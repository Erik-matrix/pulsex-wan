# 2026-10-08: taew2_2 tested (CPU 31-40 s per clip vs the full VAE's 48 min, PSNR 24.7 / 25.5 dB vs the full VAE,
# visually close) -> onto the NPU. taew2_2_super's decoder is plain Conv2d / ReLU / nearest Upsample / PixelShuffle with a
# one-frame memory (MemBlock), so the whole time axis of a tile is ONE static graph with no caches: latent tile
# [1, 13, 48, 16, 16] -> frames [1, 49, 3, 256, 256]. Full-frame does not fit the 3.5 GB graph limit (super's top
# stage expands to ~2.6 GB), so the same 8 tiles as the big VAE (rows 0/14, cols 0/12/24/36) with a linear blend.
#   python taew_npu.py check   -> CPU: tiled + blended vs whole-frame taew decode (the tiling's own cost)
#   python taew_npu.py export  -> <work>\taew_npu\taew_tile.onnx (TAEW_OUT) + inputs + torch reference
#   python taew_npu.py decode <ctx model.onnx>  -> NPU decode of the clip, PSNR vs full VAE, mp4
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import pulsex_paths as PX                                         # 10-24: every folder in one place (pulsex_paths.py)
except ImportError:                                                   # imported from outside wan22
    import os as _os
    import sys as _sys
    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    import pulsex_paths as PX

T = PX.model_file("taehv")                                            # 10-24: a copy under MODELS (taehv.py + taew2_2_super.pth, MIT); the source folder as the fallback
sys.path.insert(0, str(T))
import taehv   # noqa: E402

LAT = PX.OUT / "latent.npy"
REF = PX.OUT
TAG = "wan_480x832_f49_s25_seed1"
import os
OUT = Path(os.environ.get("TAEW_OUT", str(PX.WORK / "taew_npu")))
NAME = "taew2_2_super"
TS = 16
TILES = [(r, c) for r in (0, 14) for c in (0, 12, 24, 36)]


def starts_for(n, ov):
    """evenly spaced starts of TS-latent tiles over n latents, >= ov latents of overlap (30 / 2 -> 0 14; 52 / 4 -> 0 12 24 36)"""
    if n <= TS:
        return [0]
    k = -(-(n - TS) // (TS - ov)) + 1
    return [round(i * (n - TS) / (k - 1)) for i in range(k)]


def tiles_for(h, w):
    """10-11: the tile layout for any latent size (720p: 44 x 80 -> 3 x 7); at 30 x 52 exactly TILES"""
    return [(r, c) for r in starts_for(h, 2) for c in starts_for(w, 4)]


assert tiles_for(30, 52) == TILES


def memblocks_parallel_cat(model, x):
    """taehv.apply_model_with_memblocks_parallel with the past frame made by cat instead of pad + slice (QNN's Pad on
    these 5-D tensors was 92 % of the tile, AI Hub profile 10-09); same numbers"""
    N, T, C, H, W = x.shape
    x = x.reshape(N * T, C, H, W)
    for b in model:
        if isinstance(b, (taehv.MemBlock, taehv.SuperMemBlock)):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)
            past = torch.cat([torch.zeros_like(_x[:, :1]), _x[:, :-1]], 1).reshape(x.shape)
            x = b(x, past)
        else:
            x = b(x)
    NT, C, H, W = x.shape
    return x.view(N, NT // N, C, H, W)


class TileDecode(torch.nn.Module):
    """latent tile N T C H W -> frames N T' 3 H*16 W*16, exactly TAEHV.decode_video(parallel=True) (static shapes)"""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        y = memblocks_parallel_cat(self.m.decoder, x)
        y = self.m.postprocess_output_frames(y)
        return y[:, self.m.frames_to_trim:]


def weights(start, size, total, ov):
    w = np.ones(size, np.float32)
    ramp = (np.arange(ov, dtype=np.float32) + 0.5) / ov
    if start > 0:
        w[:ov] *= ramp
    if start + size < total:
        w[-ov:] *= ramp[::-1]
    return w


def blend(tiles_out, H=480, W=832, tiles=None):
    acc, wsum = None, np.zeros((H, W), np.float32)
    for (r0, c0), y in zip(tiles or TILES, tiles_out):                   # y [T, 3, 256, 256]
        if acc is None:
            acc = np.zeros((y.shape[0], 3, H, W), np.float32)
        ry, rx = r0 * 16, c0 * 16
        wt = weights(ry, 256, H, 32)[:, None] * weights(rx, 256, W, 64)[None, :]
        acc[:, :, ry:ry + 256, rx:rx + 256] += y * wt
        wsum[ry:ry + 256, rx:rx + 256] += wt
    return acc / wsum


def model():
    return taehv.TAEHV(checkpoint_path=str(T / (NAME + ".pth"))).eval()


def latent():
    return torch.from_numpy(np.load(LAT)).permute(1, 0, 2, 3)[None].contiguous()   # N T C H W


def psnr_frames(video, ref_dir, tag):
    from PIL import Image
    ps = []
    for k in range(video.shape[0]):
        f = (video[k].transpose(1, 2, 0) * 255).round().clip(0, 255)
        r = np.asarray(Image.open(ref_dir / ("%s_%03d.png" % (tag, k))), np.float32)
        ps.append(10 * np.log10(255.0 ** 2 / float(((f - r) ** 2).mean())))
    return np.array(ps)


def check():
    torch.set_num_threads(6)
    m, x = model(), latent()
    with torch.no_grad():
        tiles = [TileDecode(m)(x[:, :, :, r:r + TS, c:c + TS])[0].numpy() for r, c in TILES]
    v = blend(tiles)
    ps = psnr_frames(v, PX.WORK / "taew_cpu", NAME)
    print("tiled + blended vs whole-frame %s: PSNR mean %.2f dB, min %.2f (the tiling's own cost)" % (NAME, ps.mean(), ps.min()), flush=True)


def export():
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    m, x = model(), latent()
    r, c = TILES[0]
    xt = x[:, :, :, r:r + TS, c:c + TS].contiguous()
    with torch.no_grad():
        ref = TileDecode(m)(xt)
    np.save(OUT / "taew_tile_x.npy", xt.half().numpy())
    np.save(OUT / "taew_tile_y_torch.npy", ref.numpy())

    class Half(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x.float()).half()

    torch.onnx.export(Half(TileDecode(m)).eval(), (xt.half(),), str(OUT / "taew_tile.onnx"), input_names=["x"], output_names=["y"],
                      opset_version=18, dynamo=True, external_data=True)
    print("exported taew_tile.onnx: in %s fp16, out %s" % (list(xt.shape), list(ref.shape)), flush=True)


CTX = PX.MODELS / "taew_npu_ctx" / "model.onnx"


def decode_latent(lat, ctx=CTX):
    """latent [48, T, 30, 52] (normalized, as the DiT leaves it) -> frames [T', 3, 480, 832] in [0, 1], NPU (8 tiles)"""
    import onnxruntime as ort
    x = np.ascontiguousarray(np.asarray(lat, np.float32).transpose(1, 0, 2, 3)[None]).astype(np.float16)
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sess = None
    try:
        t0 = time.time()
        sess = ort.InferenceSession(str(ctx), so, providers=["QNNExecutionProvider"],
                                    provider_options=[{"backend_path": str(PX.QNN_HTP),
                                                       "htp_performance_mode": "burst"}])
        PX.qnn_witness("taew decoder")                                 # 10-25: which QNN files this session really uses
        tiles = [sess.run(None, {"x": np.ascontiguousarray(x[:, :, :, r:r + TS, c:c + TS])})[0][0].astype(np.float32) for r, c in TILES]
        print("taew on the NPU: %d tiles, %.1f s (session included)" % (len(tiles), time.time() - t0), flush=True)
    finally:
        del sess
    return np.clip(blend(tiles), 0, 1)


def decode(ctx):
    import onnxruntime as ort
    x = latent().half().numpy()
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sess = None
    ts = []
    try:
        t0 = time.time()
        sess = ort.InferenceSession(str(ctx), so, providers=["QNNExecutionProvider"],
                                    provider_options=[{"backend_path": str(PX.QNN_HTP),
                                                       "htp_performance_mode": "burst"}])
        PX.qnn_witness("taew decoder")
        print("NPU session %.1f s" % (time.time() - t0), flush=True)
        tiles = []
        for k, (r, c) in enumerate(TILES):
            t0 = time.perf_counter()
            y = sess.run(None, {"x": np.ascontiguousarray(x[:, :, :, r:r + TS, c:c + TS])})[0]
            ts.append(time.perf_counter() - t0)
            if k == 0:
                ref = np.load(OUT / "taew_tile_y_torch.npy")
                print("first tile vs torch: rel L2 %.2e" % (np.linalg.norm(y.astype(np.float32) - ref) / np.linalg.norm(ref)), flush=True)
            tiles.append(y[0].astype(np.float32))
    finally:
        del sess
        print("NPU session released", flush=True)
    v = np.clip(blend(tiles), 0, 1)
    print("NPU: %d tile calls, %.2f s total (%.3f s each)" % (len(ts), sum(ts), np.mean(ts)), flush=True)
    ps = psnr_frames(v, REF, TAG)
    print("NPU taew vs full Wan VAE: PSNR mean %.2f dB, min %.2f" % (ps.mean(), ps.min()), flush=True)
    from PIL import Image
    d = OUT / "frames"
    d.mkdir(exist_ok=True)
    for k in range(v.shape[0]):
        Image.fromarray((v[k].transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)).save(d / ("taew_npu_%03d.png" % k))
    mp4 = OUT / "taew_npu.mp4"
    subprocess.run([str(PX.FFMPEG), "-y", "-hide_banner", "-loglevel", "error", "-framerate", "24", "-i",
                    str(d / "taew_npu_%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(mp4)], check=True)
    print("wrote", mp4, flush=True)


if __name__ == "__main__":
    {"check": check, "export": export}.get(sys.argv[1], lambda: decode(Path(sys.argv[2])))()
