# 2026-10-11 night (goal: Wan 2.2 speed): the big Wan2.2 VAE decoder for wan_pipeline.py --vae npu_full, NPU only on
# the main thread. wan_vae_npu_decode.py (kept as the reference tool) spent ~25 s before its first NPU call (diffusers
# import 13 s, the CPU VAE 7.3 s) and wrote PNGs + an mp4 that the pipeline read back and wrote again. Here:
#  * the main thread needs only post_quant_conv (a 1x1x1 conv, 48 x 48) - its weights from a small cache beside the VAE
#    (keyed to the VAE file's size + content hash since 10-24), unpatchify written out (diffusers' own lines) - and starts the NPU at once
#  * a thread meanwhile imports diffusers, loads the CPU VAE and decodes tile 0, frames 0-2: the known-healthy control,
#    checked after the NPU (frames 0-1 and frame 2 of tile 0; it also checks the cached post_quant_conv against the
#    VAE's own) - a failed control writes no video
#  * the frames go back as ONE uint8 array [T, H, W, 3] (.npy); the pipeline writes PNGs and the mp4 once
# Graphs: the frames-0-1 graph (wan_vae_npu_first.py) + the step graph (wan_vae_npu.py), both WAN_PERM3 contexts.
# NPU: sessions released in finally.
#   python wan_vae_npu_fast.py <step ctx> <first ctx> <latent.npy> <out.npy>
import json
import os
import sys
import threading
import time
from pathlib import Path

# 10-25: numpy's OpenBLAS reserves per-thread buffers at import (~0.22 GB private on 8 cores); numpy does no BLAS
# work here (only the checks' norms), torch has its own BLAS - one thread. Set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wan_vae                                                        # noqa: E402  (cheap: no diffusers at import)
import pulsex_paths as PX                                             # noqa: E402  (10-24: every folder in one place)

TS = 16                                                               # tile side in latent pixels (256 px out) = wan_vae_npu.TS
TILES = [(r, c) for r in (0, 14) for c in (0, 12, 24, 36)]             # 480 x 832 (= wan_vae_npu.TILES); any size: tiles_for()
MIN_OV = (2, 2)                                                       # the overlap (latent px) at least: rows, columns - 480p: = TILES
# 10-11 night: was (2, 4) - at 720p 7 columns, 21 tiles, 1.53x the picture's area decoded. (2, 2): 6 columns (3 latents =
# 48 px of overlap), 18 tiles; 5 s 720p 855 -> 741 s, the frames 55.9 dB from the 21-tile ones (worst 54.9), the
# difference 0.37 / 255 in the blend zones (tools/seam_check_1011.py). 480p keeps its 8 tiles.
if os.environ.get("WAN_VAE_MIN_OV"):                                  # 10-11 night: e.g. "2,2" - 720p in 18 tiles instead of 21 (being measured)
    MIN_OV = tuple(int(v) for v in os.environ["WAN_VAE_MIN_OV"].split(","))


def starts_for(n_lat, min_ov):
    """10-11 (720p): tile starts along one axis - as few 16-tiles as cover n_lat with at least min_ov overlap, evenly
    spread. 30 -> 0, 14 and 52 -> 0, 12, 24, 36 (= TILES); 44 -> 0, 14, 28; 80 -> 0, 13, 26, 38, 51, 64 (min_ov 2; with 4: 7 columns)"""
    if n_lat <= TS:
        return [0]
    n = -(-(n_lat - TS) // (TS - min_ov)) + 1
    return [int(round(i * (n_lat - TS) / (n - 1))) for i in range(n)]


def tiles_for(h_lat, w_lat):
    return [(r, c) for r in starts_for(h_lat, MIN_OV[0]) for c in starts_for(w_lat, MIN_OV[1])]


def axis_weights(starts, i, total_px):
    """the blend of tile i along one axis (16 latent px = 256 px out): linear ramps over the ACTUAL overlap with each
    neighbour - the two tiles sharing an overlap ramp over the same pixels, so they sum to 1 there"""
    s = starts[i] * 16
    w = np.ones(TS * 16, np.float32)
    if i > 0:
        ov = (starts[i - 1] + TS) * 16 - s
        w[:ov] *= (np.arange(ov, dtype=np.float32) + 0.5) / ov
    if i + 1 < len(starts):
        ov = s + TS * 16 - starts[i + 1] * 16
        w[-ov:] *= ((np.arange(ov, dtype=np.float32) + 0.5) / ov)[::-1]
    return w
PQ = PX.MODELS / "Wan2.2_VAE-Q4_0.post_quant_conv.npz"
# 10-12: the known-healthy control as a GOLDEN tile. The CPU control (tile 0 frames 0-2
# of THIS latent, ~110 s and ~6 GB beside the NPU) made the step phase wait 51-77 s in every 720p film (it waits so the
# two do not stack in memory). Now the CPU reference of one fixed real tile (a 720p film's latent, tile 0, frames 0-2;
# the old control's very code) is computed ONCE (--make-golden) and keyed to the VAE file; every run feeds that tile
# through the same two NPU sessions as the film (2 calls) and compares before any video is written. WAN_VAE_CTL=cpu =
# the old CPU control; a missing or stale golden falls back to it.
GOLD = PX.MODELS / "Wan2.2_VAE-Q4_0.npu_golden_tile.npz"
QNN = PX.qnn_options()


KEY_SPAN = 16 << 20                                                   # 10-24: bytes hashed at each end of the VAE file


def vae_key(path=None):
    """10-24: the VAE file's CONTENT key - its size + SHA-256 of its first and last 16 MiB (32 MiB of the 398 MB read).
    Was [size, int(mtime)]: a downloaded or copied VAE gets a new time, so its caches were refused (sys.exit)."""
    import hashlib
    p = Path(path) if path else wan_vae_vae_path()
    n = p.stat().st_size
    h = hashlib.sha256(b"pulsex-wan-vae-key-v2|%d|" % n)
    with open(p, "rb") as f:
        h.update(f.read(KEY_SPAN))
        if n > KEY_SPAN:
            f.seek(max(KEY_SPAN, n - KEY_SPAN))
            h.update(f.read())
    return np.array("v2:%d:%s" % (n, h.hexdigest()))


def legacy_key(path=None):
    """the key before 10-24: [size, int(mtime)] (kept only to recognise - and migrate - the old caches)"""
    st = (Path(path) if path else wan_vae_vae_path()).stat()
    return np.array([st.st_size, int(st.st_mtime)], np.int64)


def key_ok(stored, path=None):
    """a cache's key against the VAE file: the content key; an old [size, mtime] key only while it still matches this
    file (as before 10-24) - then --migrate-keys rewrites it in place"""
    stored = np.asarray(stored)
    if stored.dtype.kind == "U":
        return np.array_equal(stored, vae_key(path))
    if np.array_equal(stored, legacy_key(path)):
        print("note: a VAE cache still has the old [size, mtime] key - python wan_vae_npu_fast.py --migrate-keys", flush=True)
        return True
    return False


def migrate_keys(files=None, drop=("src",)):
    """10-24: each cache's old [size, mtime] key -> the content key, IN PLACE, only while the old key still matches this
    VAE file. Every other array is written back byte-identical (checked on the re-read file before it replaces the old
    one); the golden tile's "src" field (a private path) is dropped."""
    new = vae_key()
    for f in (files or (PQ, GOLD)):
        f = Path(f)
        with np.load(f) as g:
            arrs = {k: g[k] for k in g.files}
        k = arrs["key"]
        if k.dtype.kind == "U":
            print("%s: already the content key (%s)" % (f.name, "matches" if np.array_equal(k, new) else "DOES NOT MATCH this VAE"))
            continue
        if not np.array_equal(k, legacy_key()):
            raise SystemExit("%s: its old key %s is not this VAE file's %s - not migrated (make it again)" % (f, k, legacy_key()))
        keep = {n: a for n, a in arrs.items() if n != "key" and n not in drop}
        tmp = f.with_name(f.stem + ".migrating.npz")
        np.savez(tmp, **keep, key=new)
        with np.load(tmp) as b:
            for n, a in keep.items():
                assert b[n].dtype == a.dtype and b[n].shape == a.shape and b[n].tobytes() == a.tobytes(), (f, n)
            assert np.array_equal(b["key"], new) and set(b.files) == set(keep) | {"key"}, f
        os.replace(tmp, f)
        print("%s: key %s -> %s; arrays %s byte-identical%s" % (f.name, k.tolist(), str(new), sorted(keep),
              "; dropped " + ", ".join(d for d in drop if d in arrs) if any(d in arrs for d in drop) else ""), flush=True)


def wan_vae_vae_path():
    return PX.MODELS / "Wan2.2_VAE-Q4_0.gguf"


def weights(start, size, total, ov):
    w = np.ones(size, np.float32)
    ramp = (np.arange(ov, dtype=np.float32) + 0.5) / ov
    if start > 0:
        w[:ov] *= ramp
    if start + size < total:
        w[-ov:] *= ramp[::-1]
    return w


def unpatchify(x, p=2):                                              # diffusers' unpatchify, written out
    b, cp, f, h, w = x.shape
    c = cp // (p * p)
    return x.view(b, c, p, p, f, h, w).permute(0, 1, 4, 5, 3, 6, 2).contiguous().view(b, c, f, h * p, w * p)


def session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    s = ort.InferenceSession(str(path), so, providers=["QNNExecutionProvider"], provider_options=QNN)
    PX.qnn_witness("VAE decoder, " + Path(path).parent.name)          # 10-25: which QNN files this session really uses
    return s


def commit_free_gb():
    """10-12 night: the system's commit limit minus what is committed (Task Manager's "Committed x / y"), GB"""
    import ctypes
    import ctypes.wintypes as wt

    class PERF(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
                    ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
                    ("PageSize", ctypes.c_size_t), ("HandleCount", wt.DWORD), ("ProcessCount", wt.DWORD), ("ThreadCount", wt.DWORD)]
    p = PERF()
    p.cb = ctypes.sizeof(p)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb):
        return 1e9
    return (p.CommitLimit - p.CommitTotal) * p.PageSize / 2 ** 30


MEM_LOG = os.environ.get("WAN_MEM_LOG", "0") == "1"                   # 10-25: phase memory lines (off by default)


def memlog(what):
    if MEM_LOG:
        import psutil
        print("memory: VAE private %.2f GB, system commit %.2f GB free - %s" % (psutil.Process().memory_info().private / 2 ** 30,
              commit_free_gb(), what), flush=True)


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main():
    step_ctx, first_ctx, lat_path, out_path = (Path(a) for a in sys.argv[1:5])
    T0 = time.time()
    torch.set_num_threads(6)
    lat = torch.from_numpy(np.load(lat_path))[None]
    cfg = json.loads(wan_vae.CFG.read_text())
    z = lat * torch.tensor(cfg["latents_std"]).view(1, 48, 1, 1, 1) + torch.tensor(cfg["latents_mean"]).view(1, 48, 1, 1, 1)
    pq = np.load(PQ) if PQ.exists() else None
    if pq is None or not key_ok(pq["key"]):
        sys.exit("post_quant_conv cache %s missing or not this VAE's - run: python wan_vae_npu_fast.py --make-pq" % PQ)
    Wq, bq = torch.from_numpy(pq["w"]), torch.from_numpy(pq["b"])
    x = torch.einsum("oc,bcthw->bothw", Wq, z) + bq.view(1, -1, 1, 1, 1)
    Fl, H, W = x.shape[2], x.shape[3] * 16, x.shape[4] * 16
    Tn = 1 + 4 * (Fl - 1)
    rows, cols = starts_for(x.shape[3], MIN_OV[0]), starts_for(x.shape[4], MIN_OV[1])
    TILES = [(r, c) for r in rows for c in cols]                       # this picture's tiles (10-11: any size)
    print("%d x %d: %d tiles (rows %s, columns %s)" % (W, H, len(TILES), rows, cols), flush=True)
    memlog("startup (imports, latent, post_quant_conv)")
    ctl = {}

    def control():                                                    # the known-healthy control, beside the NPU
        try:
            t0 = time.time()
            import wan_vae_npu as wv
            import decode_latent
            assert wv.TS == TS and tiles_for(30, 52) == wv.TILES, "the tile layout no longer gives wan_vae_npu's 480 x 832 tiles"
            wv.install(True)
            decode_latent.conv3d_as_conv2d()
            vae = wan_vae.load_gguf_decoder(wv.VAE, lazy=True)         # 10-11 night: decoder only; 10-12: its Q4_0 weights dequantised per layer as it runs
            pw = vae.post_quant_conv.weight.detach().reshape(48, 48)
            ctl["pq"] = max(float((pw - Wq).abs().max()), float((vae.post_quant_conv.bias.detach() - bq).abs().max()))
            with torch.no_grad():
                xv = vae.post_quant_conv(z)
                ctl["x"] = rel(x.numpy(), xv.numpy())
                r0, c0 = TILES[0]
                xt = xv[:, :, :, r0:r0 + TS, c0:c0 + TS].contiguous()
                vae.clear_cache()
                fc = vae._feat_map
                outs = [vae.decoder(xt[:, :, i:i + 1], feat_cache=fc, feat_idx=[0], first_chunk=(i == 0)) for i in range(2)]
                ctl["y2"] = vae.decoder(xt[:, :, 2:3], feat_cache=fc, feat_idx=[0], first_chunk=False).numpy()
            ctl["y01"] = torch.cat(outs, 2).numpy()
            ctl["t"] = time.time() - t0
            # 10-12 morning: the decoder and its caches held ~1.2 GB until a later garbage collection (the model has
            # reference cycles) - the first tile of the step phase ran on top of it; let it go now
            vae.clear_cache()
            del vae, fc, outs, xt, xv
            __import__("gc").collect()
        except BaseException as e:                                     # reported on the main thread
            ctl["err"] = repr(e)

    gold = None
    if os.environ.get("WAN_VAE_CTL", "golden") != "cpu" and GOLD.exists():
        g = np.load(GOLD)
        if key_ok(g["key"]):
            gold = {k: g[k] for k in g.files}
        else:
            print("golden tile %s is not this VAE's - the CPU control instead" % GOLD, flush=True)
    th = threading.Thread(target=control, daemon=True)
    if gold is None:
        th.start()
        print("startup %.1f s; control (CPU, tile 0 frames 0-2) running beside the NPU" % (time.time() - T0), flush=True)
    else:
        print("startup %.1f s; control = the golden tile (%s) through the same NPU sessions" % (time.time() - T0, GOLD.name), flush=True)
    npu0, first = {}, []
    gold_caches = None
    # 10-11 (720p: 21 tiles x 234 MB of frames-0-1 caches = 4.9 GB held in RAM, the decoder at 7.5 GB, the page file
    # grown to 12 GB): above 8 tiles (or WAN_VAE_SPILL=1) each tile's caches go to a file on C: and come back one tile at
    # a time; the files are removed as they are read (and in finally). 480p keeps them in RAM as before.
    spill = len(TILES) > 8 or os.environ.get("WAN_VAE_SPILL") == "1"
    spill_dir = PX.WORK / "wan_vae_spill"
    shapes = []
    if spill:
        spill_dir.mkdir(parents=True, exist_ok=True)
        print("frames-0-1 caches to %s, one tile at a time" % spill_dir, flush=True)
    sess = None
    t0 = time.time()
    try:
        sess = session(first_ctx)
        memlog("frames-0-1 session made")
        inn = sess.get_inputs()[0].name
        on = [o.name for o in sess.get_outputs()]
        if gold is not None:                                           # the golden tile first, its caches kept for its frame 2
            res = sess.run(on, {inn: np.ascontiguousarray(gold["x"][:, :, 0:2])})
            npu0["g01"] = res[0].astype(np.float32)
            gold_caches = res[1:]
            if spill and os.environ.get("WAN_VAE_IOBIND", "1") == "1" and not os.environ.get("WAN_VAE_CALIB_DIR"):   # 10-25: to the file too
                shapes = [(c.shape, c.dtype) for c in gold_caches]
                gf = spill_dir / "golden.bin"
                with open(gf, "wb") as fh:
                    for c in gold_caches:
                        fh.write(np.ascontiguousarray(c).tobytes())
                gold_caches = gf
            del res
        for ti, (r0, c0) in enumerate(TILES):
            xt = x[:, :, :, r0:r0 + TS, c0:c0 + TS].contiguous().numpy().astype(np.float16)
            res = sess.run(on, {inn: np.ascontiguousarray(xt[:, :, 0:2])})
            out01 = res[0].astype(np.float32)
            if (r0, c0) == TILES[0]:
                npu0["y01"] = out01
            caches = res[1:]
            if spill:
                shapes = [(c.shape, c.dtype) for c in caches]
                f = spill_dir / ("tile_%02d.bin" % ti)
                with open(f, "wb") as fh:
                    for c in caches:
                        fh.write(np.ascontiguousarray(c).tobytes())
                caches = f
            del res
            first.append((xt, torch.from_numpy(out01), caches))
    finally:
        del sess
    t_first = time.time() - t0
    memlog("frames-0-1 phase done, its session released")
    # 10-12 night: short of commit? Let the CPU control (~6 GB) finish before the step phase's session and acc land on it
    # forced: VAE peak 9.48 -> 6.89 GB, system -4.3 GB, +37 s, bit-identical. 10-12 morning: the default was "under 8 GB
    # free commit" - but the commit LIMIT follows the page file's size, and in the next 5 s 720p film the wait never came
    # (no "waited" line) and the VAE phase rose to 21.9 GB. Now always (1e9); WAN_VAE_CTL_FREE_GB=0 = never, as before.
    ctl_free = float(os.environ.get("WAN_VAE_CTL_FREE_GB", "1e9"))
    if gold is None and th.is_alive() and ctl_free > 0:
        free = commit_free_gb()
        if free < ctl_free:
            tw = time.time()
            th.join()
            print("waited %.1f s for the CPU control before the step phase (free commit %.1f GB < %.1f GB)" % (time.time() - tw, free, ctl_free), flush=True)
    # 10-12: acc is a band one tile high that moves down the picture; finished rows go straight into the uint8 video
    BAND = TS * 16
    acc = np.zeros((3, Tn, BAND, W), np.float32)                       # 0.48 GB at 720p 5 s (was the whole 1.31 GB)
    wsum = np.zeros((1, 1, BAND, W), np.float32)
    video = np.empty((Tn, H, W, 3), np.uint8)
    band0 = [rows[0] * 16]

    def finish(a, b):                                                  # rows [a, b) are final: the same ops as before
        ia, ib = a - band0[0], b - band0[0]
        v = acc[:, :, ia:ib]
        np.divide(v, wsum[:, :, ia:ib], out=v)
        np.clip(v, -1, 1, out=v)
        v += 1
        v *= 127.5
        np.clip(v, 0, 255, out=v)
        video[:, a:b] = v.astype(np.uint8).transpose(1, 2, 3, 0)
    sess = None
    calib_dir = Path(os.environ["WAN_VAE_CALIB_DIR"]) if os.environ.get("WAN_VAE_CALIB_DIR") else None   # 10-12: off by default
    if calib_dir is not None:
        calib_dir.mkdir(parents=True, exist_ok=True)
        calib_every = max(1, len(TILES) * (Fl - 2) // int(os.environ.get("WAN_VAE_CALIB_N", "8")))
    t0 = time.time()
    try:
        sess = session(step_ctx)
        memlog("step session made")
        # 10-12: a 4D step graph (wan_vae_step4d.py: frames as the batch, caches [2, C, H, W]) - told by its input's rank.
        # The frames-0-1 graph stays rank 5: its caches are converted here once per tile; x [1, 48, 16, 16]; y comes back
        # [4, 12, 128, 128] and goes on as [1, 12, 4, 128, 128].
        xshape = list(sess.get_inputs()[0].shape)
        four = len(xshape) == 4
        nhwc = four and xshape[-1] == 48                                # 10-12: NHWC boundary ([1, 16, 16, 48], caches [2, H, W, C])
        ncache = len(shapes) if spill else len(first[0][2])
        onames = (["y"] + ["n%02d" % k for k in range(ncache)]) if four else (["output_0"] + ["output_%d" % (k + 1) for k in range(ncache)])
        if nhwc:
            c_in = lambda cs: [np.ascontiguousarray(c[0].transpose(1, 2, 3, 0)) for c in cs]
            x_in = lambda xt, i: np.ascontiguousarray(xt[:, :, i].transpose(0, 2, 3, 1))
            y_out = lambda y: np.ascontiguousarray(y.transpose(3, 0, 1, 2))[None]
        elif four:
            c_in = lambda cs: [np.ascontiguousarray(c[0].transpose(1, 0, 2, 3)) for c in cs]
            x_in = lambda xt, i: np.ascontiguousarray(xt[:, :, i])
            y_out = lambda y: np.ascontiguousarray(y.transpose(1, 0, 2, 3))[None]
        else:
            c_in = lambda cs: cs
            x_in = lambda xt, i: np.ascontiguousarray(xt[:, :, i:i + 1])
            y_out = lambda y: y
        # 10-13 night: a step graph with fp32 inputs (the w8a8 QDQ model) gets them cast; its caches come back fp32 and stay
        if sess.get_inputs()[0].type == "tensor(float)":
            c_in0, x_in0 = c_in, x_in
            c_in = lambda cs: [c.astype(np.float32) for c in c_in0(cs)]
            x_in = lambda xt, i: x_in0(xt, i).astype(np.float32)
        print("step graph: %s, %s inputs" % ("4D NHWC (wan_vae_step4d.py)" if nhwc else "4D NCHW (wan_vae_step4d.py)" if four else "rank 5",
              sess.get_inputs()[0].type), flush=True)
        # 10-25: IOBinding - the caches in two fixed buffer sets (A in, B out, swapped per call), y in one; ORT writes
        # the outputs into B. The same graph, inputs and order as sess.run(): the same bytes (checked at 720p)
        use_io = calib_dir is None and os.environ.get("WAN_VAE_IOBIND", "1") == "1"
        if use_io:
            npt = {"tensor(float16)": np.float16, "tensor(float)": np.float32}
            ins = {a.name: a for a in sess.get_inputs()}
            outm = {a.name: a for a in sess.get_outputs()}
            for k in range(ncache):
                ci, co = ins["c%02d" % k], outm[onames[k + 1]]
                assert list(ci.shape) == list(co.shape) and ci.type == co.type, (k, ci.shape, co.shape, ci.type, co.type)
            bufs = [[np.empty(ins["c%02d" % k].shape, npt[ins["c%02d" % k].type]) for k in range(ncache)] for _ in range(2)]
            ybuf = np.empty(outm[onames[0]].shape, npt[outm[onames[0]].type])
            if nhwc:                                                   # one cache as the step graph takes it: c_in's
                c_view = lambda c: c[0].transpose(1, 2, 3, 0)          # layout change, without its copy
            elif four:
                c_view = lambda c: c[0].transpose(1, 0, 2, 3)
            else:
                c_view = lambda c: c
            io = sess.io_binding()
            print("step phase: IOBinding, %d caches x 2 sets of %.0f MB" % (ncache, sum(b.nbytes for b in bufs[0]) / 2 ** 20), flush=True)

            def run_io(xa, A, B):                                       # one call: x + A in, B + ybuf out
                io.clear_binding_inputs()
                io.clear_binding_outputs()
                io.bind_cpu_input("x", xa)
                for k in range(ncache):
                    io.bind_cpu_input("c%02d" % k, A[k])
                    io.bind_output(onames[k + 1], "cpu", 0, B[k].dtype, list(B[k].shape), B[k].ctypes.data)
                io.bind_output(onames[0], "cpu", 0, ybuf.dtype, list(ybuf.shape), ybuf.ctypes.data)
                sess.run_with_iobinding(io)
                return y_out(ybuf)
        if gold is not None and use_io:                                # the golden tile's frame 2: the step graph's first call
            if isinstance(gold_caches, Path):                         # 10-25: spilled - one cache at a time into A
                with open(gold_caches, "rb") as fh:
                    for k, (shp, dt) in enumerate(shapes):
                        c = np.fromfile(fh, dtype=dt, count=int(np.prod(shp))).reshape(shp)
                        np.copyto(bufs[0][k], c_view(c))
                        del c
                gold_caches.unlink()
            else:
                for k, c in enumerate(gold_caches):
                    np.copyto(bufs[0][k], c_view(c))
            npu0["g2"] = run_io(x_in(gold["x"], 2), bufs[0], bufs[1]).astype(np.float32)
            del gold_caches
            memlog("golden frame 2 done")
        elif gold is not None:                                         # the golden tile's frame 2: the step graph's first call
            feeds = {"x": x_in(gold["x"], 2), **{"c%02d" % k: c for k, c in enumerate(c_in(gold_caches))}}
            npu0["g2"] = y_out(sess.run(onames[:1], feeds)[0]).astype(np.float32)
            del feeds, gold_caches
            memlog("golden frame 2 done")
        for ti,((r0, c0), (xt, out01, caches)) in enumerate(zip(TILES, first)):
            if use_io:
                A, B = bufs
                if spill:                                           # this tile's caches from C:, one at a time into A
                    with open(caches, "rb") as fh:
                        for k, (shp, dt) in enumerate(shapes):
                            c = np.fromfile(fh, dtype=dt, count=int(np.prod(shp))).reshape(shp)
                            np.copyto(A[k], c_view(c))
                            del c
                    caches.unlink()
                else:
                    for k, c in enumerate(caches):
                        np.copyto(A[k], c_view(c))
                first[ti] = (xt, out01, None)
                caches = None
                outs = [out01]
                for i in range(2, Fl):
                    y_i = run_io(x_in(xt, i), A, B)
                    if (r0, c0) == TILES[0] and i == 2:
                        npu0["y2"] = y_i.astype(np.float32)
                    outs.append(torch.from_numpy(y_i.astype(np.float32)))
                    A, B = B, A                                         # this call's outputs = the next call's inputs
                bufs = [A, B]
            else:                                                       # the old path (WAN_VAE_IOBIND=0, calibration)
                if spill:                                               # this tile's caches back from C:, the file gone
                    raw, off, back = caches.read_bytes(), 0, []
                    for shp, dt in shapes:
                        n = int(np.prod(shp)) * np.dtype(dt).itemsize
                        back.append(np.frombuffer(raw, dtype=dt, count=int(np.prod(shp)), offset=off).reshape(shp))
                        off += n
                    caches.unlink()
                    caches = back
                    first[ti] = (xt, out01, None)
                caches = c_in(caches)
                outs = [out01]
                for i in range(2, Fl):
                    feeds = {"x": x_in(xt, i), **{"c%02d" % k: c for k, c in enumerate(caches)}}
                    if calib_dir is not None and (ti * (Fl - 2) + (i - 2)) % calib_every == calib_every // 2:   # 10-12: w8a16 calibration
                        np.savez(calib_dir / ("t%02d_f%02d.npz" % (ti, i)), **feeds)
                    res = sess.run(onames, feeds)
                    y_i = y_out(res[0])
                    if (r0, c0) == TILES[0] and i == 2:
                        npu0["y2"] = y_i.astype(np.float32)
                    outs.append(torch.from_numpy(y_i.astype(np.float32)))
                    caches = res[1:]
            y = unpatchify(torch.cat(outs, 2))[0].numpy()               # [3, T, 256, 256]
            if ti < 2 or ti == len(TILES) - 1:
                memlog("tile %d decoded" % ti)
            ry, rx = r0 * 16, c0 * 16
            if ry != band0[0]:                                         # a new tile row: the rows above it are final
                finish(band0[0], ry)
                sh = ry - band0[0]
                keep_a, keep_w = acc[:, :, sh:].copy(), wsum[:, :, sh:].copy()   # the overlap still to come
                acc[:] = 0
                wsum[:] = 0
                acc[:, :, :BAND - sh] = keep_a
                wsum[:, :, :BAND - sh] = keep_w
                del keep_a, keep_w
                band0[0] = ry
            wt = (axis_weights(rows, rows.index(r0), H)[:, None] * axis_weights(cols, cols.index(c0), W)[None, :])[None, None]
            acc[:, :, ry - band0[0]:ry - band0[0] + 256, rx:rx + 256] += y * wt
            wsum[:, :, ry - band0[0]:ry - band0[0] + 256, rx:rx + 256] += wt
    finally:
        del sess
        io = bufs = ybuf = None                                       # 10-25: the step phase's buffers (none on the old path)
        if spill:                                                   # what a failure left behind (our own files only)
            for f in list(spill_dir.glob("tile_*.bin")) + list(spill_dir.glob("golden.bin")):
                f.unlink()
    t_step = time.time() - t0
    memlog("step phase done, its session released")
    t0 = time.time()
    if gold is not None:                                               # 10-12: the golden control (CPU reference made once)
        # 10-13 night: a LOSSY step graph (w8a8: 8.9e-3 from the CPU reference, too close to the 0.01 bar) keeps its own
        # golden frame 2 beside its context (golden_y2.npy, its NPU output for the golden tile, recorded once by
        # tools/vae_w8a8_golden_1013.py): compared with the same tight bar - a broken graph still fails it - and the CPU
        # reference too, with a loose bar (0.05) that only a gross fault passes
        # 10-25: the own-output bar 0.01 -> 1e-3. On one QNN set it is exactly 0.00e+00; a change of QNN set gave
        # 7.73e-3 and passed 0.01 (QNN_KONSEKVENT_RAPPORT_1025.md) - now it fails the run (tools/golden_bar_check_1025.py)
        own = Path(step_ctx).parent / "golden_y2.npy"
        own_y2 = np.load(own) if own.exists() else None
        checks = [("g01", "golden tile frames 0-1 on the NPU vs its CPU reference", gold["y01"], 0.01)]
        if own_y2 is None:
            checks.append(("g2", "golden tile frame 2 (step graph) vs its CPU reference", gold["y2"], 0.01))
        else:
            checks.append(("g2", "golden tile frame 2 (step graph) vs its own recorded output", own_y2, 1e-3))
            checks.append(("g2", "golden tile frame 2 (step graph, lossy) vs the CPU reference", gold["y2"], 0.05))
        for k, msg, ref_k, bar in checks:
            r = rel(npu0[k], ref_k)
            print("%s: rel L2 %.2e" % (msg, r), flush=True)
            # fixed input, so a tighter bar than the CPU control's 0.05: healthy 2.9e-4 / 4.2e-4, the tile shifted by
            # one latent pixel 0.117 / 0.114 (tools/vae_golden_sick_1012.py)
            # 10-13 night: WAN_VAE_GOLD_BAR only for measuring a lossy step graph (w8a8) through a whole film - never a default
            assert r < float(os.environ.get("WAN_VAE_GOLD_BAR", bar)), (k, r, msg)
    else:
        th.join()
        if "err" in ctl:
            sys.exit("the CPU control failed: " + ctl["err"])
        print("control %.1f s (waited %.1f s after the NPU): cached post_quant_conv vs the VAE's max|d| %.1e, x rel %.1e" % (
            ctl["t"], time.time() - t0, ctl["pq"], ctl["x"]), flush=True)
        assert ctl["pq"] == 0.0 and ctl["x"] < 1e-5, (ctl["pq"], ctl["x"])
        for k, msg in (("y01", "frames 0-1 on the NPU vs the CPU (tile 0)"), ("y2", "first call vs the CPU (frame 2, tile 0)")):
            r = rel(npu0[k], ctl[k])
            print("%s: rel L2 %.2e" % (msg, r), flush=True)
            assert r < 0.05, (k, r)
    # [T, H, W, 3]; truncated to uint8 as wan_vae_npu_decode.py's PNGs (so the two paths' frames compare bit for bit)
    finish(band0[0], H)                                                # 10-12: the last band (the rest went row by row)
    del acc, wsum
    memlog("before the video is saved")
    np.save(out_path, video)
    print("decode: NPU frames 0-1 %.1f s, NPU %.1f s (%d calls, %.2f s each), total %.1f s" % (
        t_first, t_step, len(TILES) * (Fl - 2), t_step / (len(TILES) * (Fl - 2)), time.time() - T0), flush=True)


def make_pq():
    """the post_quant_conv cache, from the VAE itself (and keyed to its file)"""
    import wan_vae_npu as wv
    vae = wan_vae.load_gguf(wv.VAE)
    assert Path(wv.VAE) == wan_vae_vae_path(), wv.VAE
    w = vae.post_quant_conv.weight.detach().reshape(48, 48).numpy()
    b = vae.post_quant_conv.bias.detach().numpy()
    np.savez(PQ, w=w, b=b, key=vae_key())
    print("wrote", PQ, w.shape, b.shape)


def make_golden(lat_path):
    """10-12: the golden control tile - tile 0, frames 0-2 of a real latent, through post_quant_conv (the cached one, as
    main() feeds the NPU, in fp16) and the CPU reference of exactly the old control (the VAE's own post_quant_conv,
    lazy decoder, frame by frame with its feature cache); keyed to the VAE file."""
    import wan_vae_npu as wv
    import decode_latent
    torch.set_num_threads(6)
    lat = torch.from_numpy(np.load(lat_path))[None]
    cfg = json.loads(wan_vae.CFG.read_text())
    z = lat * torch.tensor(cfg["latents_std"]).view(1, 48, 1, 1, 1) + torch.tensor(cfg["latents_mean"]).view(1, 48, 1, 1, 1)
    pq = np.load(PQ)
    assert key_ok(pq["key"]), "post_quant_conv cache is not this VAE's"
    x = torch.einsum("oc,bcthw->bothw", torch.from_numpy(pq["w"]), z) + torch.from_numpy(pq["b"]).view(1, -1, 1, 1, 1)
    wv.install(True)
    decode_latent.conv3d_as_conv2d()
    vae = wan_vae.load_gguf_decoder(wv.VAE, lazy=True)
    with torch.no_grad():
        xv = vae.post_quant_conv(z)
        xt = xv[:, :, 0:3, 0:TS, 0:TS].contiguous()
        vae.clear_cache()
        fc = vae._feat_map
        outs = [vae.decoder(xt[:, :, i:i + 1], feat_cache=fc, feat_idx=[0], first_chunk=(i == 0)) for i in range(2)]
        y2 = vae.decoder(xt[:, :, 2:3], feat_cache=fc, feat_idx=[0], first_chunk=False).numpy()
    y01 = torch.cat(outs, 2).numpy()
    xg = x[:, :, 0:3, 0:TS, 0:TS].contiguous().numpy().astype(np.float16)
    np.savez(GOLD, x=xg, y01=y01, y2=y2, key=vae_key())                 # 10-24: no "src" (it named a private path)
    print("wrote", GOLD, "x", xg.shape, "y01", y01.shape, "y2", y2.shape)


if __name__ == "__main__":
    if sys.argv[1] == "--make-pq":
        make_pq()
    elif sys.argv[1] == "--migrate-keys":                              # 10-24: old [size, mtime] keys -> content keys
        migrate_keys()
    elif sys.argv[1] == "--key":                                       # 10-24: print a VAE file's content key
        print(str(vae_key(sys.argv[2] if len(sys.argv) > 2 else None)))
    elif sys.argv[1] == "--make-golden":
        make_golden(sys.argv[2])
    else:
        main()
