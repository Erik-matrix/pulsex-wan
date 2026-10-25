# 2026-10-08 night (goal: the best decisions for NPU generation): the whole Wan2.2-TI2V-5B text-to-video chain around
# the 30 blocks, with the blocks behind one call so the NPU contexts (compile_blocks.py) can replace the CPU later:
#   umT5-XXL Q8_0 (umt5_check.encode) -> text_embedding (4096 -> 3072, GELU-tanh, 3072) on 512 zero-padded tokens
#   noise [48, F, H/16, W/16] -> per step: t = 1000 sigma, sinusoid(256) -> time_embedding -> e (head) and
#   time_projection -> e0 (6 x 3072, every block); patch_embedding Conv3d (1, 2, 2) -> 30 blocks -> head (LayerNorm
#   without affine, shift/scale from head.modulation + e, Linear 3072 -> 192) -> unpatchify -> flow velocity;
#   classifier-free guidance with Wan's negative prompt; Euler on the shifted flow-matching sigmas (shift 5, TI2V-5B)
#   -> latents * std + mean -> Wan2.2 VAE Q4_0 (wan_vae.load_gguf) -> frames.
# T2V only: one timestep for every token (TI2V's image mode sets t = 0 on the first frame's tokens, which our compiled
# graphs - one modulation vector per tile - do not take yet).
# The CPU blocks: the GGUF stays memory-mapped as Q4_0/Q4_1 bytes (5B params in fp32 = 20 GB, the machine has 15.6) and
# each block is dequantized with torch when it runs. CPU only, low priority, threads capped.
#   python wan_pipeline.py "prompt" [--size 480x832] [--frames 1] [--steps 25] [--guide 5] [--seed 1] [--threads 6]
import math
import os
import sys
import time
from pathlib import Path

# 10-25: numpy's OpenBLAS reserves per-thread buffers at import (~0.22 GB private on 8 cores, held through the whole
# film); numpy does no BLAS work on this path, torch has its own BLAS - one thread. Before numpy is imported; the VAE
# process inherits it.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
import torch.nn.functional as F_
import gguf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wan_block as wb                                                 # noqa: E402
import umt5_check as ut                                                # noqa: E402
import pulsex_paths as PX                                              # noqa: E402  (10-24: every folder in one place)

DIR = PX.SRC                                                          # the developer's source folder (fallbacks, --backend cpu)
_UC = PX.MODELS
UMT5 = _UC / "umt5-xxl-encoder-Q8_0-city96.gguf"   # 10-09: city96 (from the original weights, llama.cpp names) - the Python fallback reads it too
SPIECE = _UC / "spiece.model" if (_UC / "spiece.model").exists() else DIR / "spiece.model"
OUT = PX.OUT                                                          # the default --outdir; made only when the film is written (10-24)
CDIR = PX.MODELS                                                      # the files the NPU path reads
EXTRAS = CDIR / "Wan2.2-TI2V-5B-npu-extras.gguf"                     # make_slim_dit.py: everything but the blocks
VAE = CDIR / "Wan2.2_VAE-Q4_0.gguf" if (CDIR / "Wan2.2_VAE-Q4_0.gguf").exists() else DIR / "Wan2.2_VAE-Q4_0.gguf"
DIM, NB, LC = 3072, 30, 512
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
       "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
       "三条腿，背景人很多，倒着走")                                      # Wan2.2's sample_neg_prompt


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


class Q4Model:
    """the DiT GGUF memory-mapped; tensors dequantized with torch on demand (Q4_0 / Q4_1 / F16 / F32)"""
    def __init__(self, path=wb.GGUF):
        self.t = {t.name: t for t in gguf.GGUFReader(str(path)).tensors}

    def get(self, name):
        t = self.t[name]
        shape = tuple(int(s) for s in reversed(t.shape))
        T = gguf.GGMLQuantizationType
        if t.tensor_type in (T.F16, T.F32):
            return torch.from_numpy(np.asarray(t.data, dtype=np.float32).reshape(shape))
        raw = torch.from_numpy(np.ascontiguousarray(t.data).reshape(-1, 18 if t.tensor_type == T.Q4_0 else 20))
        d = raw[:, 0:2].contiguous().view(torch.float16).float()
        qs = raw[:, 2:18] if t.tensor_type == T.Q4_0 else raw[:, 4:20]
        q = torch.cat((qs & 15, qs >> 4), 1).float()                     # 32 values per block: low nibbles, then high
        if t.tensor_type == T.Q4_0:
            a = (q - 8) * d
        elif t.tensor_type == T.Q4_1:
            a = q * d + raw[:, 2:4].contiguous().view(torch.float16).float()
        else:
            raise ValueError(t.tensor_type)
        return a.reshape(shape)

    def block(self, i):
        p = "blocks.%d." % i
        w = {k[len(p):]: self.get(k).numpy() for k in self.t if k.startswith(p)}
        return wb.WanBlock(w).eval()


def sinusoid(t, dim=256):
    half = dim // 2
    s = torch.outer(torch.tensor([t], dtype=torch.float64), torch.pow(10000, -torch.arange(half, dtype=torch.float64) / half))
    return torch.cat([torch.cos(s), torch.sin(s)], 1).float()


def sigmas(steps, shift=5.0):
    s = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    return (shift * s / (1 + (shift - 1) * s)).tolist()


TEXT_CACHE = PX.TEXT_CACHE


def encode_text(prompts):
    import hashlib
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(SPIECE))
    import os
    import umt5_llama
    use_llama = umt5_llama.available() and os.environ.get("WAN_T5", "llama") != "python"
    enc_name = umt5_llama.GGUF.name if use_llama else UMT5.name
    src = None
    out = []
    for p in prompts:
        key = hashlib.sha256((enc_name + "|" + p).encode("utf-8")).hexdigest()[:24]
        f = TEXT_CACHE / (key + ".npy")
        if f.exists():
            c = torch.from_numpy(np.load(f).astype(np.float32))
            print("text cache hit %s (%d tokens): %.40s..." % (key[:8], c.shape[0], p), flush=True)
        else:
            if use_llama:
                t0e = time.time()
                c = torch.from_numpy(umt5_llama.encode(p)[:LC])
                print("umT5 via llama.cpp: %d tokens, %.1f s" % (c.shape[0], time.time() - t0e), flush=True)
            else:
                if src is None:
                    ut.Q4 = str(UMT5)
                    src = ut.Q4Src()
                ids = (sp.encode(p) + [1])[:LC]
                with torch.no_grad():
                    c = ut.encode(src, ids)
            TEXT_CACHE.mkdir(parents=True, exist_ok=True)
            c16 = c.numpy().astype(np.float16)                         # what the cache holds
            np.save(f, c16)
            # 10-25: a miss goes on with exactly what a hit reads back (was the fp32 output: a new prompt's first film
            # differed from the same prompt's next one - rel L2 4.5e-2 in the latent)
            c = torch.from_numpy(c16.astype(np.float32))
            print("text cache store %s (%d tokens): %.40s..." % (key[:8], c.shape[0], p), flush=True)
        out.append(torch.cat([c, torch.zeros(LC - c.shape[0], c.shape[1])], 0)[None])   # zero-padded to 512 like Wan
    return out


class DiT:
    def __init__(self, m):
        self.m = m
        g = m.get
        self.te = [(g("text_embedding.0.weight"), g("text_embedding.0.bias")), (g("text_embedding.2.weight"), g("text_embedding.2.bias"))]
        self.tm = [(g("time_embedding.0.weight"), g("time_embedding.0.bias")), (g("time_embedding.2.weight"), g("time_embedding.2.bias"))]
        self.tp = (g("time_projection.1.weight"), g("time_projection.1.bias"))
        self.pe = (g("patch_embedding.weight"), g("patch_embedding.bias"))      # [3072, 48, 1, 2, 2]
        self.hm = g("head.modulation").reshape(1, 2, DIM)
        self.hd = (g("head.head.weight"), g("head.head.bias"))

    def text(self, c):
        return F_.linear(wb.gelu_tanh(F_.linear(c, *self.te[0])), *self.te[1])

    def time(self, t):
        e = F_.linear(F_.silu(F_.linear(sinusoid(t), *self.tm[0])), *self.tm[1])        # [1, 3072]
        e0 = F_.linear(F_.silu(e), *self.tp).reshape(1, 6, DIM)
        return e, e0

    def patch(self, lat):
        return F_.conv3d(lat[None], *self.pe, stride=(1, 2, 2)).flatten(2).transpose(1, 2)   # [1, L, 3072]

    HEAD_CHUNK = 4096   # 10-13: the head in token chunks - its layer norm and modulation made two full
                        # [1, L, 3072] fp32 temporaries (2 x 335 MB at 720p 5 s); every token is its own row, so per chunk = the same

    def head(self, x, e, grid, n0=0, e_b=None):
        mod = self.hm + e[:, None]
        mb = self.hm + e_b[:, None] if (n0 and e_b is not None) else None   # image to video: the first frame's tokens at t = 0
        L = x.shape[1]
        out = torch.empty(1, L, self.hd[0].shape[0], dtype=x.dtype)
        for a in range(0, L, self.HEAD_CHUNK):
            b = min(L, a + self.HEAD_CHUNK)
            ln = F_.layer_norm(x[:, a:b], (DIM,), eps=1e-6)
            if mb is not None and a < n0:
                k = min(b, n0) - a                                      # the chunk's first k tokens are the picture's
                h = torch.cat([ln[:, :k] * (1 + mb[:, 1:2]) + mb[:, 0:1], ln[:, k:] * (1 + mod[:, 1:2]) + mod[:, 0:1]], 1)
            else:
                h = ln * (1 + mod[:, 1:2]) + mod[:, 0:1]
            out[:, a:b] = F_.linear(h, *self.hd)
        x = out                                                         # [1, L, 192]
        f, h, w = grid
        u = x[0].reshape(f, h, w, 1, 2, 2, 48)
        return torch.einsum("fhwpqrc->cfphqwr", u).reshape(48, f, h * 2, w * 2)


WAN_STREAM = PX.BIN / "wan-stream.exe"
VAE_NPU_CTX = CDIR / "vae_npu_ctx_near" / "model.onnx"              # the 10-08 nearest-fix context (older copies computed wrong)
if (CDIR / "vae_npu_step_p3" / "model.onnx").exists():             # 10-11: + WAN_PERM3 (rank-3 transposes): 6.88 -> 1.26 s per call
    VAE_NPU_CTX = CDIR / "vae_npu_step_p3" / "model.onnx"
# 10-12: the step graph in 4D (wan_vae_step4d.py: frames as batch, NHWC boundary, channel repeats as Gather), compiled
# locally: 1.288 -> 0.954 s per call, golden 4.21e-4 (p3 4.22e-4). WAN_VAE_STEP=p3 = the rank-5 context above.
if (CDIR / "vae_npu_step_4d" / "model.onnx").exists() and __import__("os").environ.get("WAN_VAE_STEP", "4d") != "p3":
    VAE_NPU_CTX = CDIR / "vae_npu_step_4d" / "model.onnx"
# 10-13 night, OPT-IN (a deliberate precision trade): the same 4D graph in w8a8 (tools/vae_w8a8_1013.py, local QDQ +
# saved context): 0.43 s per call (fp16 0.95); a whole 720p 5 s decode 597 -> 440 s with the context prepared in the
# run (~315 s saved-context estimate); frames vs fp16 mean 47.5 dB, worst 42.7 dB, no drift over 121 frames.
# Its golden frame 2 is its own recorded output (golden_y2.npy beside it).
# 10-13 morning: the DEFAULT since. The night's film 33.8 min (fp16 4D 38.7),
# vs the reference 40.2 dB (fp16 40.8). WAN_VAE_STEP=4d = the fp16 4D graph, =p3 = the rank-5 one.
if (CDIR / "vae_npu_step_w8a8" / "model.onnx").exists() and __import__("os").environ.get("WAN_VAE_STEP", "w8a8") == "w8a8":
    VAE_NPU_CTX = CDIR / "vae_npu_step_w8a8" / "model.onnx"
WAN_GGML_GGUF = CDIR / "Wan2.2-TI2V-5B-Q4_0-ffn2q8-ggml.gguf"           # make_ggml_gguf.py + make_ffn2_q8.py (ffn.2 Q8_0, 10-09); .hexpack beside it
if (CDIR / "Wan2.2-TI2V-5B-Q4_0-ffn2q8-ggml-slim.gguf").exists():    # 10-10: SLIM (slim_gguf.py, 0.17 GB) + its .hexpack; the full file on D:
    WAN_GGML_GGUF = CDIR / "Wan2.2-TI2V-5B-Q4_0-ffn2q8-ggml-slim.gguf"
# 10-13: --dit turbo = Wan2.2-TI2V-5B-Turbo (quanhaol, Self-Forcing/DMD: 4 steps, no CFG;
# hum-ma's city96 Q4_0, Apache-2.0) through the same chain as the base (make_ggml_gguf + make_ffn2_q8 + pack + slim,
# make_slim_dit for the extras) in models/wan22/turbo. Its own recipe: --steps 4 --guide 1 --solver euler (shift 5:
# the sigmas at 1000 / 750 / 500 / 250 warped, as DMD trained them).
DIT = (sys.argv[sys.argv.index("--dit") + 1] if "--dit" in sys.argv else "base")
if DIT == "turbo":
    EXTRAS = CDIR / "turbo" / "Wan2.2-TI2V-5B-Turbo-npu-extras.gguf"
    WAN_GGML_GGUF = CDIR / "turbo" / "Wan2.2-TI2V-5B-Turbo-Q4_0-ffn2q8-ggml-slim.gguf"
    if not WAN_GGML_GGUF.exists():                                   # before the slim step: the full file (its pack beside it)
        WAN_GGML_GGUF = CDIR / "turbo" / "Wan2.2-TI2V-5B-Turbo-Q4_0-ffn2q8-ggml.gguf"


class WanServe:
    """one resident wan-stream process (HTP0) for the whole video; closed in close() and by the engine at end of input"""
    WD = PX.WORK / "wan_stream"

    # 10-09, measured at 12090 tokens (121 frames): K/V as f16 into the flash attention and Br 768 x Bc 512 take a block from
    # 1639 to 1227 ms (self attention 1078 -> 652 ms) - K/V no longer fit VTCM resident there and are re-prepared per Q tile,
    # so both the f16 K/V (no per-tile conversion) and fewer, taller Q tiles pay. At 5070 tokens they cost instead (K/V
    # resident): wan-stream's own defaults (f32 K/V, Br 256 x Bc 1280) stay. vs torch at 12090: 1.349e-3 vs 1.313e-3.
    LONG = 8000

    def __init__(self, tokens=0):
        import subprocess
        self.WD.mkdir(parents=True, exist_ok=True)
        env = dict(__import__("os").environ, WAN_DEVICE="HTP0")
        if tokens > self.LONG:
            env.update({"WAN_KV_F16": "1", "WAN_FA_TILING": "0", "GGML_HEXAGON_FA_BR": "768", "GGML_HEXAGON_FA_BC": "512"})
            print("wan-stream: long sequence (%d tokens): f16 K/V, flash tiles 768 x 512" % tokens, flush=True)
        self.log = open(self.WD / "engine.log", "w", encoding="utf-8")
        t0 = time.time()
        self.p = subprocess.Popen([str(WAN_STREAM), "serve", str(WAN_GGML_GGUF), str(self.WD)], cwd=str(WAN_STREAM.parent),
                                  env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, text=True, bufsize=1)
        self.expect("ok ready")
        self.t = {"files": 0.0, "engine": 0.0, "read": 0.0, "copy": 0.0, "run": 0.0}
        print("wan-stream serve: ready in %.1f s" % (time.time() - t0), flush=True)

    def expect(self, head):
        line = self.p.stdout.readline().strip()
        if not line.startswith(head):
            raise SystemExit("wan-stream: wanted '%s', got '%s' (see %s)" % (head, line, self.WD / "engine.log"))
        return line

    def text(self, name, ctx):
        np.ascontiguousarray(ctx[0].numpy(), np.float32).tofile(self.WD / (name + ".f32"))
        self.p.stdin.write("text %s\n" % name)
        self.expect("ok text")

    def run(self, x, e0, name, grid, e0b=None, n0=0):
        # 10-14 (a long text-to-video run: 11 tokens x 48 channels of the LAST step came back NaN and the VAE
        # spread them through its time caches - a film of noise after frame 0, and nothing said so). A pass whose output
        # is not finite is run once more (HTP0 is not always bit-deterministic); still not finite = stop, loudly.
        for attempt in (1, 2):
            y = self._run_once(x, e0, name, grid, e0b, n0)
            bad = ~torch.isfinite(y)
            if not bool(bad.any()):
                return y
            tok = torch.nonzero(bad[0].any(-1)).flatten()
            print("WARNING: wan-stream pass %s gave %d non-finite values in %d tokens (first %s) - attempt %d of 2" % (
                name, int(bad.sum()), tok.numel(), tok[:8].tolist(), attempt), flush=True)
        raise SystemExit("wan-stream: pass %s non-finite twice - stopped before the VAE could turn it into noise" % name)

    def _run_once(self, x, e0, name, grid, e0b=None, n0=0):
        t0 = time.time()
        np.ascontiguousarray(x[0].numpy(), np.float32).tofile(self.WD / "x.f32")
        np.ascontiguousarray(e0[0].numpy(), np.float32).tofile(self.WD / "e0.f32")
        if n0:
            np.ascontiguousarray(e0b[0].numpy(), np.float32).tofile(self.WD / "e0b.f32")
        t1 = time.time()
        self.p.stdin.write("pass %d %d %d %s %d\n" % (*grid, name, n0))
        f = self.expect("ok pass").split()
        t2 = time.time()
        y = torch.from_numpy(np.fromfile(self.WD / "y.f32", np.float32).reshape(x.shape))
        self.t["files"] += (t1 - t0) + (time.time() - t2)
        self.t["engine"] += t2 - t1
        for k, v in zip(("read", "copy", "run"), f[3:6]):
            self.t[k] += float(v)
        return y

    def close(self):
        if self.p.poll() is None:
            try:
                self.p.stdin.write("quit\n")
                self.p.stdin.flush()
                self.p.wait(timeout=30)
            except Exception:
                self.p.kill()
        self.log.close()
        for f in ("x.f32", "e0.f32", "e0b.f32", "y.f32", "c.f32", "u.f32"):
            (self.WD / f).unlink(missing_ok=True)


def encode_image(path, H, W):
    """the picture, cover-cropped to W x H, through the full Wan2.2 VAE encoder (F16 weights, CPU) -> [48, 1, H/16, W/16],
    normalised as the decoder un-normalises (lat * std + mean)"""
    from PIL import Image
    import wan_vae
    t0 = time.time()
    im = Image.open(path).convert("RGB")
    k = max(W / im.width, H / im.height)
    im = im.resize((max(W, round(im.width * k)), max(H, round(im.height * k))), Image.LANCZOS)
    l, t = (im.width - W) // 2, (im.height - H) // 2
    im = im.crop((l, t, l + W, t + H))
    x = torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None, :, None]   # [1, 3, 1, H, W]
    global PIC_X
    PIC_X = x                                                            # 10-11: for the end check (frame 0 vs the picture)
    cfg = __import__("json").loads(wan_vae.CFG.read_text())
    mean = torch.tensor(cfg["latents_mean"]).view(1, 48, 1, 1, 1)
    std = torch.tensor(cfg["latents_std"]).view(1, 48, 1, 1, 1)
    # one encoder graph per picture size: vae_npu_enc = 480 x 832, vae_npu_enc_<H>x<W> for others (10-11: 704 x 1280)
    enc_ctx = CDIR / ("vae_npu_enc" if (H, W) == (480, 832) else "vae_npu_enc_%dx%d" % (H, W)) / "model.onnx"
    if enc_ctx.exists() and "--enc-cpu" not in sys.argv:
        # 10-11: the encoder graph on the NPU (wan_vae_enc_npu.py: one picture needs no caches); patchify on the CPU,
        # written out here (diffusers' own three lines) - importing diffusers' autoencoder module cost 12.9 s
        import onnxruntime as ort
        p = cfg.get("patch_size", 2)
        b_, c_, f_, h_, w_ = x.shape
        xp = (x.view(b_, c_, f_, h_ // p, p, w_ // p, p).permute(0, 1, 6, 4, 2, 3, 5).contiguous()
              .view(b_, c_ * p * p, f_, h_ // p, w_ // p).half().numpy())
        so = ort.SessionOptions()
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        sess = None
        t1 = time.time()
        try:
            sess = ort.InferenceSession(str(enc_ctx), so, providers=["QNNExecutionProvider"],
                                        provider_options=PX.qnn_options())
            PX.qnn_witness("VAE encoder")                               # 10-25: which QNN files this session really uses
            t2 = time.time()
            mu = torch.from_numpy(sess.run(None, {sess.get_inputs()[0].name: xp})[0].astype(np.float32))
        finally:
            del sess
        where = "NPU: picture %.1f s, session %.1f s, run %.1f s" % (t1 - t0, t2 - t1, time.time() - t2)
    else:
        vae = wan_vae.load_f16()
        with torch.no_grad():
            mu = vae.encode(x).latent_dist.mode()
        del vae
        # 10-11 (at 720p: committed up to 21 GB at the start): the CPU VAE (fp32) and the 720p encode stayed held -
        # 5.8 GB for ~2 min after the encode, until Python's collector happened to run (sampled: 5.9 -> 1.1 GB at 14:54:51)
        import gc
        gc.collect()
        where = "CPU"
    z = ((mu - mean) / std)[0]
    print("picture %s -> first latent frame %s, VAE encoder (%s) %.1f s" % (Path(path).name, list(z.shape), where, time.time() - t0), flush=True)
    return z


PIC_X = None


def main():
    prompt = sys.argv[1]
    H, W = (int(v) for v in arg("--size", "480x832").split("x"))
    frames, steps, guide, seed = arg("--frames", 1), arg("--steps", 25), arg("--guide", 5.0), arg("--seed", 1)
    nblocks = arg("--blocks", NB)                                       # < 30 only for a smoke test
    backend = arg("--backend", "cpu")   # cpu | ggml (wan-stream on the NPU)
    if backend not in ("cpu", "ggml"):
        raise SystemExit("--backend must be cpu or ggml (the NPU runs through wan-stream: --backend ggml)")
    torch.set_num_threads(arg("--threads", 6))
    # 10-24: no OUT.mkdir here - the folder is made after --outdir, when the latent is written (a machine without D: fell here)
    prefetch = None
    if arg("--solver", "euler") == "unipc":                              # 10-11: diffusers' package import costs 7.3 s - start it
        import threading                                                 # now, beside the text / model / picture work; JOINED before
        prefetch = threading.Thread(target=lambda: __import__("diffusers.schedulers.scheduling_unipc_multistep"), daemon=True)
        prefetch.start()                                                 # use (diffusers swaps in a lazy module at the end of its
                                                                         # __init__: an import racing it saw a half-made package)
    t0 = time.time()
    # 10-16: WAN_NEG_EXTRA adds to the negative prompt (only with guidance, i.e. not Turbo) - e.g. "hands near the face"
    ctx_c, ctx_u = encode_text([prompt, NEG + __import__("os").environ.get("WAN_NEG_EXTRA", "")])
    print("umT5 Q8_0: %.1f s" % (time.time() - t0), flush=True)
    assert DIT == "base" or (backend == "ggml" and EXTRAS.exists() and WAN_GGML_GGUF.exists()), (
        "--dit %s needs --backend ggml and its files (%s, %s)" % (DIT, EXTRAS, WAN_GGML_GGUF))   # never the base weights silently
    m = Q4Model(EXTRAS if backend in ("npu", "ggml") and EXTRAS.exists() else wb.GGUF)   # ggml: the blocks live in wan-stream
    dit = DiT(m)
    with torch.no_grad():
        ctx_c, ctx_u = dit.text(ctx_c), dit.text(ctx_u)
        Fl, Hl, Wl = (frames - 1) // 4 + 1, H // 16, W // 16
        grid = (Fl, Hl // 2, Wl // 2)
        cos, sin = wb.rope_tables(*grid)
        lat = torch.randn(48, Fl, Hl, Wl, generator=torch.Generator().manual_seed(seed))
        image = arg("--image", "")
        z_img, n0, e_b, e0_b = None, 0, None, None
        if image:
            assert backend in ("ggml", "cpu"), "--image needs --backend ggml or cpu (the QNN contexts take one t only)"
            z_img = encode_image(image, H, W)
            # 10-13 (hum-ma's Turbo note: "Multiplier 0.8 applied to latent will help with oversaturation"): in ComfyUI's
            # image-to-video graph the latent node holds the picture's latent (the rest is zeros), so it scales the
            # picture's frame - which the whole film follows. 1.0 = as before.
            z_img = z_img * arg("--image-latent-scale", 1.0)
            lat[:, :1] = z_img
            n0 = grid[1] * grid[2]                                       # the first latent frame's tokens come first
            e_b, e0_b = dit.time(0.0)                                    # clean: t = 0
        sg = sigmas(steps)
        solver, cfg_until = arg("--solver", "euler"), arg("--cfg-until", 1.0)
        cfg_rescale, cfg_rescale_ch = arg("--cfg-rescale", 0.0), "--cfg-rescale-channel" in sys.argv   # 10-11: off by default
        # 10-11: TeaCache-style step skipping (patch_pipeline_step_cache_1011.py) - 1-based steps; the residual per branch
        skip_steps = {int(v) for v in arg("--skip-steps", "").split(",") if v.strip()}
        cache_log, res_prev, xin_prev = "--cache-log" in sys.argv, {}, None
        # 10-23 (OPTIMERING_1023 #1): CFG cache - on these 1-based CFG steps only the conditional pass runs and the guidance
        # term reuses the last computed D = vc - vu; empty = off (the old path)
        cfg_reuse = {int(v) for v in arg("--cfg-reuse", "").split(",") if v.strip()}
        d_last = None
        keep_res = bool(skip_steps) or cache_log                         # 10-11: the residuals cost 2 x 335 MB at 720p 5 s
        res16 = __import__("os").environ.get("WAN_RES_F16", "1") == "1"  # 10-12: fp16 (480p 2 s: 59.5 / 54.6 dB vs f32)
        if solver == "unipc":
            if prefetch is not None:
                prefetch.join()
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
            sched = UniPCMultistepScheduler(num_train_timesteps=1000, prediction_type="flow_prediction", use_flow_sigmas=True,
                                            flow_shift=5.0, solver_order=2)
            sched.set_timesteps(steps)
            tlist = [float(t) for t in sched.timesteps]
            print("UniPC timesteps:", " ".join("%.0f" % t for t in tlist), flush=True)
        npu = None
        srv = None
        if backend == "ggml":
            srv = WanServe(grid[0] * grid[1] * grid[2])
            srv.text("c", ctx_c)
            srv.text("u", ctx_u)
        tph = {"patch": 0.0, "head": 0.0, "solver": 0.0}
        # 10-13: every step's time embedding up front (the same call per t), then the text / time
        # weights go - 285 MB fp32 (time_projection alone 226 MB) the pipeline held through all steps and the VAE
        temb = [dit.time(tlist[s] if solver == "unipc" else 1000 * sg[s]) for s in range(steps)]
        dit.te = dit.tm = dit.tp = None
        __import__("gc").collect()
        mem_log = __import__("os").environ.get("WAN_MEM_LOG", "0") == "1"   # 10-13: the pipeline's private bytes per step
        if mem_log:
            import psutil
            _pp = psutil.Process()
            print("memory: pipeline private %.2f GB before the steps" % (_pp.memory_info().private / 2 ** 30), flush=True)
        print("latent %s, %d tokens, %d steps (%s), guide %.1f for the first %d" % (list(lat.shape), grid[0] * grid[1] * grid[2],
              steps, solver, guide, int(round(cfg_until * steps))), flush=True)
        for s in range(steps):
            ts = time.time()
            t_s = tlist[s] if solver == "unipc" else 1000 * sg[s]
            g = guide if s < round(cfg_until * steps) else 1.0                 # the CFG schedule
            e, e0 = temb[s]
            tq = time.time()
            xc = xu = dit.patch(lat)
            tph["patch"] += time.time() - tq
            skipped = False
            use_d = False                                                     # 10-23: the guidance from d_last, not a u pass
            if srv is not None:                                               # our own ggml stack on HTP0, resident
                x_in = xc
                use_d = bool(cfg_reuse) and g != 1.0 and d_last is not None and (
                    (s + 1) in cfg_reuse or ((s + 1) in skip_steps and "c" in res_prev))
                if (s + 1) in skip_steps and "c" in res_prev and (g == 1.0 or use_d or "u" in res_prev):
                    def plus(r16):                                            # this step's input + the last residual, in
                        y = torch.empty_like(x_in)                            # token chunks (10-13: the full .float() was
                        for a in range(0, x_in.shape[1], 4096):               # one more 335 MB tensor; elementwise = same)
                            y[:, a:a + 4096] = x_in[:, a:a + 4096] + r16[:, a:a + 4096].float()
                        return y
                    xc = plus(res_prev["c"])
                    if g != 1.0 and not use_d:
                        xu = plus(res_prev["u"])
                    skipped = True
                else:
                    xc = srv.run(xc, e0, "c", grid, e0_b, n0)
                    if g != 1.0 and not use_d:
                        xu = srv.run(xu, e0, "u", grid, e0_b, n0)
                    for b_, xo in ((("c", xc), ("u", xu)) if g != 1.0 and not use_d else (("c", xc),)) if keep_res else ():
                        if res16 and not cache_log:
                            # 10-13: straight into fp16 in token chunks - the full fp32 difference was a
                            # third 335 MB tensor at 720p 5 s; elementwise, so the same numbers
                            r16 = torch.empty(xo.shape, dtype=torch.float16)
                            for a in range(0, xo.shape[1], 4096):
                                r16[:, a:a + 4096] = (xo[:, a:a + 4096] - x_in[:, a:a + 4096]).half()
                            res_prev[b_] = r16
                            continue
                        r_ = xo - x_in
                        if cache_log and b_ in res_prev:
                            print("cache %s step %d: residual change rel L1 %.4f, input change rel L1 %.4f" % (
                                b_, s + 1, float((r_ - res_prev[b_]).abs().mean() / res_prev[b_].abs().mean()),
                                float((x_in - xin_prev).abs().mean() / xin_prev.abs().mean())), flush=True)
                        res_prev[b_] = r_.half() if res16 else r_
                        r_ = None
                    xo = None                                             # the loop variable kept the last pass alive
                    xin_prev = x_in if cache_log else None
                x_in = None                                               # 10-11: and a one-pass step's xu is only an alias
                if g == 1.0:                                              # of this step's input (335 MB at 720p 5 s)
                    xu = None
            else:
                e0t = e0
                if n0:                                                    # per token: the first frame at t = 0
                    e0t = e0.reshape(1, 1, 6, DIM).repeat(1, xc.shape[1], 1, 1).clone()
                    e0t[:, :n0] = e0_b.reshape(1, 1, 6, DIM)
                for i in range(nblocks):
                    blk = m.block(i)
                    xc = blk(xc, e0t, ctx_c, cos, sin)
                    if g != 1.0:
                        xu = blk(xu, e0t, ctx_u, cos, sin)
                    del blk
            tq = time.time()
            vc = dit.head(xc, e, grid, n0, e_b)
            xc = None                                                     # 10-13: let the 335 MB passes go before the next patch
            v = vc
            if g != 1.0 and use_d:                                            # 10-23: CFG cache - vc + (g - 1)*D_last
                xu = None
                v = vc + (g - 1.0) * d_last
            elif g != 1.0:
                vu = dit.head(xu, e, grid, n0, e_b)
                xu = None
                v = vu + g * (vc - vu)
                if cfg_reuse:
                    d_last = vc - vu
                if cfg_rescale > 0:
                    # 10-11 (text to video came out cyan and blown; the cast grows with CFG): CFG rescale (Lin et al.
                    # 2023) - the guided velocity brought back to the spread of the conditional one, blended by phi.
                    # "channel": the spread per latent channel (the cast is a colour = channel matter)
                    dims = tuple(range(1, v.dim())) if cfg_rescale_ch else tuple(range(v.dim()))
                    v = cfg_rescale * (v * (vc.std(dim=dims, keepdim=True) / v.std(dim=dims, keepdim=True).clamp(min=1e-8))) + (1 - cfg_rescale) * v
            tr = time.time()
            if solver == "unipc":
                lat = sched.step(v, tlist[s], lat).prev_sample
            else:
                lat = lat + (sg[s + 1] - sg[s]) * v
            if z_img is not None:
                lat[:, :1] = z_img                                        # the picture stays the first frame
            tph["head"] += tr - tq
            tph["solver"] += time.time() - tr
            wait = ""
            print("step %d/%d t %.0f %s: %.1f s%s%s" % (s + 1, steps, t_s, ("CFG (u reused)" if use_d else "CFG") if g != 1.0
                                                     else "1 pass", time.time() - ts, wait,
                                                     " (skipped: cached residual)" if skipped else ""), flush=True)
            if mem_log:
                print("memory: pipeline private %.2f GB after step %d" % (_pp.memory_info().private / 2 ** 30, s + 1), flush=True)
        if srv is not None:
            print("wan-stream totals: " + ", ".join("%s %.1f s" % kv for kv in srv.t.items()) + " | python: " +
                  ", ".join("%s %.1f s" % kv for kv in tph.items()), flush=True)
            srv.close()
        # 10-11 night: the DiT's state is not needed past here - the VAE runs in its own process; let it go first
        xc = xu = x_in = xin_prev = v = vc = vu = m = dit = sched = r_ = xo = d_last = None   # r_ / xo: the residual loop's last tensors (335 MB each)
        res_prev.clear()
        temb = cos = sin = ctx_c = ctx_u = None
        __import__("gc").collect()
        if mem_log:
            print("memory: pipeline private %.2f GB before the VAE" % (_pp.memory_info().private / 2 ** 30), flush=True)
        name = arg("--name", "latent")
        out = Path(arg("--outdir", str(OUT)))
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / (name + "_latent.npy"), lat.numpy())
        if not bool(torch.isfinite(lat).all()):                           # 10-14: never decode a latent with NaN/inf
            raise SystemExit("the latent has %d non-finite values - not decoded (saved as %s)" % (
                int((~torch.isfinite(lat)).sum()), out / (name + "_latent.npy")))
        vae_kind = arg("--vae", "full")                                  # full = Wan2.2 VAE (CPU), taew = taew2_2_super (NPU), taew_cpu = taew on the CPU (any length)
        if vae_kind == "npu_full":
            # 10-10: the big Wan2.2 VAE - frames 0-1 on the CPU, the rest on the NPU (wan_vae_npu_decode.py); its own
            # known-healthy control runs on THIS latent. npu_full = 49 frames at 480x832 only (the context's tiles).
            import subprocess
            vdir = out / "_vae_npu"
            t0v = time.time()
            fast_first = CDIR / "vae_npu_first_p3" / "model.onnx"
            if fast_first.exists():
                # 10-11: wan_vae_npu_fast.py - NPU at once, the CPU control in a thread, the frames back as one array
                # 10-12: the 327 MB frame array (720p 5 s) handed over on C:, not through the film's folder on the
                # SD card (91 MB/s each way) - beside the VAE's own cache spill
                vtmp = PX.WORK
                vtmp.mkdir(parents=True, exist_ok=True)
                vnpy = vtmp / (name + "_video_u8.npy")
                r = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).with_name("wan_vae_npu_fast.py")), str(VAE_NPU_CTX),
                                    str(fast_first), str(out / (name + "_latent.npy")), str(vnpy)], capture_output=True, text=True, errors="replace")
                if r.returncode != 0:
                    raise SystemExit("the big VAE on the NPU failed:\n" + r.stdout[-1500:] + r.stderr[-1500:])
                print("\n".join(l for l in r.stdout.splitlines() if l.startswith(("control", "frames 0-1", "first call", "decode:", "waited", "golden", "startup", "step graph", "step phase", "QNN", "WARNING", "memory:"))
                                 or " tiles (rows " in l), flush=True)   # 10-12 night: + the wait for the control and the layout
                y = torch.from_numpy(np.load(vnpy).astype(np.float32)).permute(3, 0, 1, 2) / 127.5 - 1
                vnpy.unlink()
                print("big VAE on the NPU: %d frames, %.1f s" % (y.shape[1], time.time() - t0v), flush=True)
            else:
                raise SystemExit("the big VAE needs its NPU contexts: %s is missing" % fast_first)
        elif vae_kind == "taew":
            import taew_npu
            y = torch.from_numpy(taew_npu.decode_latent(lat.numpy())).permute(1, 0, 2, 3) * 2 - 1   # [3, T, H, W] in -1..1
        elif vae_kind == "taew_cpu":                                     # 10-09: any frame count (the NPU taew is compiled for 13 latent frames)
            import taew_npu
            tm = taew_npu.model()
            x5 = lat.permute(1, 0, 2, 3)[None].contiguous()               # [1, T, 48, H, W]
            t0v = time.time()
            lay = taew_npu.tiles_for(lat.shape[2], lat.shape[3])          # 10-11: the latent's own layout (720p: 21 tiles), not 480x832's
            tiles = [taew_npu.TileDecode(tm)(x5[:, :, :, r:r + taew_npu.TS, c:c + taew_npu.TS])[0].numpy() for r, c in lay]
            y = torch.from_numpy(np.clip(taew_npu.blend(tiles, lat.shape[2] * 16, lat.shape[3] * 16, lay), 0, 1)).permute(1, 0, 2, 3) * 2 - 1
            print("taew on the CPU: %d tiles, %d frames, %.1f s" % (len(tiles), y.shape[1], time.time() - t0v), flush=True)
        else:
            import wan_vae
            cfg = __import__("json").loads(wan_vae.CFG.read_text())
            mean = torch.tensor(cfg["latents_mean"]).view(1, 48, 1, 1, 1)
            std = torch.tensor(cfg["latents_std"]).view(1, 48, 1, 1, 1)
            vae = wan_vae.load_gguf(VAE)
            if lat.shape[1] > 1:                                        # 10-08: whole frames asked conv3d for 22 GB (4 frames
                vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,   # per latent frame at 480x832); spatial
                                  tile_sample_stride_height=192, tile_sample_stride_width=192)   # tiles with blended overlap
            y = vae.decode(lat[None] * std + mean).sample.clamp(-1, 1)[0]   # [3, T, H, W]
    from PIL import Image
    tag = "%s_%dx%d_f%d_s%d_seed%d" % (name, H, W, frames, steps, seed)
    u8 = ((y.permute(1, 2, 3, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)   # [T, H, W, 3], as the PNGs always were
    no_png = "--no-png" in sys.argv                                      # 10-11: the window (wan_gui.cpp) wants only the mp4
    if not no_png:
        from concurrent.futures import ThreadPoolExecutor               # 10-11: 6 writers (49 PNGs 6.7 -> 4.5 s on D:)
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(lambda k: Image.fromarray(u8[k]).save(out / ("%s_%03d.png" % (tag, k))), range(u8.shape[0])))
    mp4 = out / (tag + ".mp4")
    if y.shape[1] > 1:
        import subprocess
        ff = [str(PX.FFMPEG), "-y", "-hide_banner", "-loglevel", "error", "-framerate", "24"]
        enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(mp4)]
        if no_png:
            subprocess.run(ff + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (u8.shape[2], u8.shape[1]), "-i", "-"] + enc,
                           input=np.ascontiguousarray(u8).tobytes(), check=True)
        else:
            subprocess.run(ff + ["-i", str(out / (tag + "_%03d.png"))] + enc, check=True)
        print("wrote", mp4, flush=True)
    print("wrote %s (%d frames%s), total %.1f s" % (out / tag, y.shape[1], "" if no_png else ", PNGs", time.time() - t0), flush=True)
    if PIC_X is not None:
        # 10-11: the chain's own check - a broken encoder (or decoder) cannot give back the picture as frame 0
        f0 = ((y[:, 0].numpy() + 1) * 127.5).clip(0, 255)
        p0 = ((PIC_X[0, :, 0].numpy() + 1) * 127.5).clip(0, 255)
        mse = float(((f0 - p0) ** 2).mean())
        ps = 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)
        print("frame 0 vs the picture: %.2f dB%s" % (ps, "" if ps >= 20 else "  ⚠ BELOW 20 dB - the encoder or decoder is wrong"), flush=True)
    preview = arg("--preview", "")
    if preview:                                                          # 10-11: small frames for the window to play
        pw = arg("--preview-w", 416)
        ph = int(round(u8.shape[1] * pw / u8.shape[2]))
        with open(preview, "wb") as f:
            for k in range(u8.shape[0]):
                f.write(np.asarray(Image.fromarray(u8[k]).resize((pw, ph), Image.BOX), np.uint8).tobytes())
        # tab-separated: the paths can hold spaces (10-11: "%LOCALAPPDATA%\PulseX Wan" split a space-separated line)
        print("RESULT\t%s\t%s\t%d\t%d\t%d" % (mp4, preview, pw, ph, u8.shape[0]), flush=True)


if __name__ == "__main__":
    main()
