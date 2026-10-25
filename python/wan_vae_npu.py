# 2026-10-08: the VAE on the NPU. The Wan2.2 VAE decoder (diffusers AutoencoderKLWan, Q4_0 weights in
# fp16 on the NPU) as ONE static graph per latent frame, in the steady state: from latent frame 2 on, every causal conv's
# cache (the last 2 frames it saw) has a fixed shape, so the graph takes  latent frame + all caches  and returns
# 4 output frames (12 ch, 240x416 before unpatchify) + the new caches. Latent frames 0 and 1 (1 + 4 of the 49 frames;
# their caches are None / "Rep" / 1 frame) run on the CPU, as do post_quant_conv (1x1x1), unpatchify and clamp.
# QNN rewrites, each checked against diffusers' own code in the same run (known-healthy control):
#  * DupUp3D: rank-8 view/permute -> pixel_shuffle with the frames as batch + rank-5 permutes (QNN: rank <= 5)
#  * WanResample upsample3d: rank-6 reshape/stack -> rank 5 (batch 1)
#  * WanRMS_norm: x / max|x| before F.normalize (scale-invariant, exact) - HTP saturates x^2 at 2^17 (LTX VAE, 09-29)
#   python wan_vae_npu.py prep [latent.npy]   -> caches + reference + ONNX in <work>\vae_npu (or --out FOLDER)
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wan_vae                                                        # noqa: E402
import pulsex_paths as PX                                             # noqa: E402  (10-24: every folder in one place)
from diffusers.models.autoencoders import autoencoder_kl_wan as m     # noqa: E402

OUT = PX.WORK / "vae_npu"
if "--out" in sys.argv:
    OUT = Path(sys.argv[sys.argv.index("--out") + 1])
VAE = PX.MODELS / "Wan2.2_VAE-Q4_0.gguf"
PERM3 = os.environ.get("WAN_PERM3", "0") == "1"                       # 10-10: the big permutes as rank-3 transposes (exact)
RMS_POW = os.environ.get("WAN_RMS_POW", "0") == "1"                   # 10-12 night: for a w8a16 graph (off: as before)
REP_CONV = os.environ.get("WAN_REP_CONV", "0") == "1"                 # 10-12 night: for a w8a16 graph (off: as before)
# 10-12 (the frames-0-1 graph after the 4D step): the channel repeats of DupUp3D and the nearest x2 upsample as ONE Gather
# (index_select, index built in numpy = a constant) instead of repeat_interleave's rank-5 unsqueeze / expand / reshape -
# in the step graph that alone took a call from 1.22 to 0.95 s (wan_vae_step4d.py STEP4D_REP=gather). Off: as before.
REP_GATHER = os.environ.get("WAN_REP_GATHER", "0") == "1"


def rep_gather(x, r):
    """x.repeat_interleave(r, dim=1) as index_select with a constant index"""
    idx = torch.from_numpy(np.arange(x.shape[1] * r, dtype=np.int64) // r)
    return x.index_select(1, idx)
TS = 16                                                               # tile side in latent pixels (256 px out)
TILES = [(r, c) for r in (0, 14) for c in (0, 12, 24, 36)]             # 2 x 4 equal tiles over 30 x 52
ORIG = {"ups": m.WanUpsample.forward, "dup": m.DupUp3D.forward, "res": m.WanResample.forward, "rms": m.WanRMS_norm.forward, "conv": m.WanCausalConv3d.forward}


def dup_rank5(self, x, first_chunk=False):
    """DupUp3D without rank 8: out[c, t*ft+a, h*fs+i, w*fs+j] = rep(x)[(((c*ft+a)*fs+i)*fs+j), t, h, w] (batch 1)"""
    if REP_CONV:   # 10-12 night (w8a16): QNN rejects repeat_interleave's Unsqueeze/Expand/Reshape in 16-bit; a 1x1x1 conv
        c = x.shape[1]  # with 0/1 weights is the same copy, exact in float (tools/qdq16_op_probe_1012.py: accepted)
        wn = np.zeros((c * self.repeats, c, 1, 1, 1), np.float32)       # built in numpy: a CONSTANT in the export (made
        wn[np.arange(c * self.repeats), np.arange(c * self.repeats) // self.repeats] = 1.0   # with torch ops it was traced
        x = F.conv3d(x, torch.from_numpy(wn).to(x.dtype))               # as ops - a computed weight QNN will not take)
    elif REP_GATHER:
        x = rep_gather(x, self.repeats)
    else:
        x = x.repeat_interleave(self.repeats, dim=1)
    _, C, T, H, W = x.shape
    ft, fs, co = self.factor_t, self.factor_s, self.out_channels
    y = x[0].transpose(0, 1)                                             # [T, C, H, W]
    if fs > 1:
        y = F.pixel_shuffle(y, fs)                                       # [T, co*ft, H*fs, W*fs]
    if PERM3:   # 10-10: the same move as a rank-3 transpose (exact): probe B 2030 -> 73.8 ms on the device (permute_probe_1010.py)
        y = y.reshape(T, co, ft * H * fs * W * fs).transpose(0, 1).reshape(1, co, T * ft, H * fs, W * fs)
    else:
        y = y.reshape(T, co, ft, H * fs, W * fs).permute(1, 0, 2, 3, 4).reshape(1, co, T * ft, H * fs, W * fs)
    if first_chunk:
        y = y[:, :, ft - 1:]
    return y


def interleave2(x, c, t, h, w):
    """time_conv's [1, 2c, t, h, w] -> [1, c, 2t, h, w], out[i, 2t + k] = x[k*c + i, t]; WAN_PERM3=1 (10-10) as a rank-3
    transpose of [2, c*t, h*w] - the same order (c, t, k, h, w), exact - instead of the rank-5 permute"""
    if PERM3:
        return x.reshape(2, c * t, h * w).transpose(0, 1).reshape(1, c, t * 2, h, w)
    return x.reshape(2, c, t, h, w).permute(1, 2, 0, 3, 4).reshape(1, c, t * 2, h, w)


def res_rank5(self, x, feat_cache=None, feat_idx=[0]):
    """WanResample.forward with the upsample3d time interleave in rank 5 (batch 1); everything else as diffusers"""
    b, c, t, h, w = x.size()
    if self.mode == "upsample3d" and feat_cache is not None and feat_cache[feat_idx[0]] is not None \
            and not isinstance(feat_cache[feat_idx[0]], str):
        idx = feat_idx[0]
        cache_x = x[:, :, -m.CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2:
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
        x = self.time_conv(x, feat_cache[idx])                            # [1, 2c, t, h, w]
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
        x = interleave2(x, c, t, h, w)                                   # (k, c) -> c, t*2 + k
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        return x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
    return ORIG["res"](self, x, feat_cache, feat_idx)


def rms_maxscaled(self, x):
    d = 1 if self.channel_first else -1
    mx = x.abs().amax(dim=d, keepdim=True).clamp(min=1e-12)
    if RMS_POW:    # 10-12 night (w8a16): F.normalize's ReduceL2 (its square root) is rejected in 16-bit; after the max
        v = x / mx  # scaling ||v|| >= 1, so v * (sum v^2) ** -0.5 is the same normalisation (qdq16_op_probe_1012.py: accepted)
        return v * (v * v).sum(dim=d, keepdim=True).pow(-0.5) * self.scale * self.gamma + self.bias
    return F.normalize(x / mx, dim=d) * self.scale * self.gamma + self.bias


def ups_shuffle(self, x):
    """WanUpsample (nearest-exact x2): HTP's Resize got it wrong (up0 probe, rel L2 0.71); out[c, 2h+i, 2w+j] = x[c, h, w]
    = pixel_shuffle of every channel repeated 4 times (the ops DupUp3D already runs correctly on the NPU)"""
    sf = self.scale_factor if isinstance(self.scale_factor, tuple) else (self.scale_factor, self.scale_factor)
    if x.dim() != 4 or tuple(float(v) for v in sf) != (2.0, 2.0) or self.mode not in ("nearest", "nearest-exact"):
        return ORIG["ups"](self, x)
    # 10-08: pixel_shuffle was right but cost 3.39 -> 5.90 s per call (a 4x-channel tensor at every upsampler); the
    # variants are exact too (CPU 0.0): expand+reshape (broadcast copies, rank <= 5) and plain asymmetric nearest
    v = os.environ.get("WAN_UPS", "shuffle")
    if v == "expand":
        N, C, H, W = x.shape
        return (x.unsqueeze(-1).expand(N, C, H, W, 2).reshape(N, C, H, 2 * W)
                .unsqueeze(3).expand(N, C, H, 2, 2 * W).reshape(N, C, 2 * H, 2 * W))
    if v == "nearest":
        return F.interpolate(x, scale_factor=2, mode="nearest")
    if REP_GATHER:
        return F.pixel_shuffle(rep_gather(x, 4), 2)
    return F.pixel_shuffle(x.repeat_interleave(4, dim=1), 2)


def install(rewrites=True):
    m.WanUpsample.forward = ups_shuffle if rewrites else ORIG["ups"]
    m.DupUp3D.forward = dup_rank5 if rewrites else ORIG["dup"]
    m.WanResample.forward = res_rank5 if rewrites else ORIG["res"]
    m.WanRMS_norm.forward = rms_maxscaled if rewrites else ORIG["rms"]


class Step(torch.nn.Module):
    """one steady-state latent frame: (x [1, 48ch post_quant, 1, 30, 52], *caches) -> (out [1, 12, 4, 240, 416], *caches)"""
    def __init__(self, decoder, nslots, used, frames_only=False):
        super().__init__()
        self.dec, self.nslots, self.used, self.frames_only = decoder, nslots, list(used), frames_only

    def forward(self, x, *caches):
        fc = [None] * self.nslots
        for k, c in zip(self.used, caches):
            fc[k] = c
        out = self.dec(x, feat_cache=fc, feat_idx=[0])
        assert all(fc[k] is None for k in range(self.nslots) if k not in self.used)
        if self.frames_only:
            return out
        return (out, *[fc[k] for k in self.used])


class Step16(torch.nn.Module):
    """Step with fp16 at the graph boundary (10-08, the CCTV lesson: the native type in and out): 468 MB of fp32
    caches crossed into and out of every call; inside it stays the same fp32 graph that QNN compiles to fp16 anyway"""
    def __init__(self, step):
        super().__init__()
        self.step = step

    def forward(self, x, *caches):
        outs = self.step(x.float(), *[c.float() for c in caches])
        return outs.half() if isinstance(outs, torch.Tensor) else tuple(o.half() for o in outs)


def prep(latent_path):
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    lat = torch.from_numpy(np.load(latent_path))[None]                   # [1, 48, 13, 30, 52]
    cfg = json.loads(wan_vae.CFG.read_text())
    z = lat * torch.tensor(cfg["latents_std"]).view(1, 48, 1, 1, 1) + torch.tensor(cfg["latents_mean"]).view(1, 48, 1, 1, 1)
    vae = wan_vae.load_gguf(VAE)
    dec = vae.decoder
    with torch.no_grad():
        x = vae.post_quant_conv(z)
        # control: the rank-5 rewrites against diffusers' code on latent frames 0..2 (small crop, CPU)
        xc = x[:, :, :3, :8, :8]
        outs = {}
        for rw in (False, True):
            install(rw)
            vae.clear_cache()
            fc, ys = vae._feat_map, []
            for i in range(3):
                ys.append(dec(xc[:, :, i:i + 1], feat_cache=fc, feat_idx=[0], first_chunk=(i == 0)))
            outs[rw] = torch.cat(ys, 2)
        d = float((outs[True] - outs[False]).abs().max())
        print("control: rank-5 rewrites + max-scaled norm vs diffusers, 3 latent frames 8x8: max |diff| %.2e" % d, flush=True)
        assert d < 1e-3, d
        # the real frames 0 and 1 of tile (0, 0) (CPU, rewrites on) -> the caches that latent frame 2 sees
        install(True)
        r0, c0 = TILES[0]
        x = x[:, :, :, r0:r0 + TS, c0:c0 + TS].contiguous()
        import decode_latent
        decode_latent.conv3d_as_conv2d()                                  # conv3d -> conv2d: CPU speed for frames 0-2
        t0 = time.time()
        vae.clear_cache()
        fc = vae._feat_map
        for i in range(2):
            dec(x[:, :, i:i + 1], feat_cache=fc, feat_idx=[0], first_chunk=(i == 0))
        used = [k for k, c in enumerate(fc) if c is not None]
        assert all(isinstance(fc[k], torch.Tensor) for k in used), [type(c) for c in fc]
        caches = [fc[k].contiguous() for k in used]
        print("cache slots: %d of %d used" % (len(used), len(fc)), flush=True)
        print("frames 0-1 on CPU: %.1f s; %d caches, %.1f MB fp32: %s" % (time.time() - t0, len(caches),
              sum(c.numel() for c in caches) * 4 / 1e6, sorted({tuple(c.shape) for c in caches})), flush=True)
        K = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 1
        step = Step(dec, len(fc), used).eval()
        x2 = x[:, :, 2:2 + K].contiguous()
        t0 = time.time()
        ref = step(x[:, :, 2:3].contiguous(), *[c.clone() for c in caches])
        if K > 1:                                                         # frames 2..2+K-1 one by one (== K at once)
            ys, cc = [ref[0]], list(ref[1:])
            for i in range(3, 2 + K):
                r = step(x[:, :, i:i + 1].contiguous(), *cc)
                ys.append(r[0])
                cc = list(r[1:])
            ref = (torch.cat(ys, 2),)
            step = Step(dec, len(fc), used, frames_only=True).eval()
        print("reference on CPU for %d latent frame(s): %.1f s, out %s" % (K, time.time() - t0, list(ref[0].shape)), flush=True)
    names = ["x"] + ["c%02d" % k for k in range(len(caches))]
    fp16io = "--fp16io" in sys.argv
    io = (lambda t: t.half()) if fp16io else (lambda t: t)
    np.save(OUT / "wan_vae_step_x.npy", io(x2).numpy())
    for k, c in enumerate(caches):
        np.save(OUT / ("wan_vae_step_c%02d.npy" % k), io(c).numpy())
    np.save(OUT / "wan_vae_step_y_torch.npy", ref[0].numpy())
    for k, c in enumerate(ref[1:]):
        np.save(OUT / ("wan_vae_step_ref_c%02d.npy" % k), c.numpy())
    onames = ["y"] + (["n%02d" % k for k in range(len(caches))] if len(ref) > 1 else [])
    if "--conv2d" not in sys.argv:
        m.WanCausalConv3d.forward = ORIG["conv"]                          # plain conv3d (HTP has Conv3d)
    else:
        print("exporting with conv3d as a sum of conv2d (no Conv3d in the graph)", flush=True)
    t0 = time.time()
    if fp16io:
        print("exporting with fp16 inputs/outputs (fp32 inside)", flush=True)
    torch.onnx.export(Step16(step).eval() if fp16io else step, (io(x2), *[io(c) for c in caches]), str(OUT / "wan_vae_step.onnx"), input_names=names,
                      output_names=onames, opset_version=18, dynamo=True, external_data=True)
    print("exported wan_vae_step.onnx %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else str(PX.OUT / "latent.npy"))
