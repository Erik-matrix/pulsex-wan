# 2026-10-09: umT5-XXL through llama.cpp's T5 encoder (llama-embedding, CPU), the way the LTX
# chain runs its T5 (ltxvideo.py t5_encode: ~8 s with --load-mode dio): city96's umT5 GGUF in llama.cpp naming (per-layer
# relative attention bias = umT5), --pooling none -> one 4096-vector per token incl. </s>, --attention non-causal
# (without it the prompt is ignored - LTX 09-29). Replaces the Python streaming encoder (~70 s, weight-read bound).
#   python umt5_llama.py "prompt"     -> tokens, time, and per-token cosine vs the Python encoder (umt5_check)
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import pulsex_paths as PX                                         # 10-24: every folder in one place (pulsex_paths.py)
except ImportError:                                                   # imported from outside wan22
    import os as _os
    import sys as _sys
    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    import pulsex_paths as PX

BIN = PX.BIN
GGUF = PX.MODELS / "umt5-xxl-encoder-Q8_0-city96.gguf"
BELOW_NORMAL = 0x4000


def available():
    return (BIN / "llama-embedding.exe").exists() and GGUF.exists()


def encode(prompt, threads=6, device="none"):
    """prompt -> [n_tokens, 4096] float32 (encoder output incl. the final norm, </s> included)"""
    # 10-24: under WORK, made if missing (was a fixed folder made only by __main__ - FileNotFoundError elsewhere)
    with tempfile.TemporaryDirectory(dir=str(PX.work_dir())) as d:
        pf = Path(d) / "prompt.txt"
        pf.write_text(prompt, encoding="utf-8")
        cmd = [str(BIN / "llama-embedding.exe"), "-m", str(GGUF), "-f", str(pf), "--pooling", "none",
               "--embd-normalize", "-1", "--embd-output-format", "array", "--embd-separator", "<#nosplit#>",
               "-c", "512", "-ngl", "0" if device == "none" else "99", "--device", device, "-t", str(threads), "--attention", "non-causal",
               # 10-25: mmap, not dio - the same embedding bytes, 1.36 GB less private memory (7.84 -> 6.48 GB), +0.6 s
               "--load-mode", os.environ.get("WAN_T5_LOAD", "mmap")] + (["--override-tensor", "token_embd=CPU"] if device != "none" else [])   # 1.12 GB > HTP max buffer 1 GiB
        log = Path(d) / "log.txt"
        with open(log, "w", encoding="utf-8", errors="replace") as lf:
            p = subprocess.run(cmd, cwd=str(BIN), stdout=subprocess.PIPE, stderr=lf, creationflags=BELOW_NORMAL)
        if p.returncode != 0:
            raise RuntimeError("llama-embedding failed:\n" + log.read_text(encoding="utf-8", errors="replace")[-2000:])
        return np.array(json.loads(p.stdout.decode("utf-8")), dtype=np.float32).reshape(-1, 4096)


if __name__ == "__main__":
    PX.work_dir()                                                     # (encode() makes it itself since 10-24)
    prompt = sys.argv[1] if len(sys.argv) > 1 else "A waterfall in a wide northern river, white foam swirling in the pool below the falls, spruce forest all around."
    for k in range(2):                                                   # cold, then warm
        t0 = time.time()
        e = encode(prompt)
        print("llama.cpp umT5: %d tokens, %.1f s, std %.4f" % (e.shape[0], time.time() - t0, e.std()), flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import sentencepiece as spm
    import umt5_check as ut
    sp = spm.SentencePieceProcessor(model_file=str(PX.MODELS / "spiece.model"))
    ids = sp.encode(prompt) + [1]
    ut.Q4 = str(PX.MODELS / "umt5-xxl-encoder-Q8_0.gguf")
    torch.set_num_threads(6)
    with torch.no_grad():
        ref = ut.encode(ut.Q4Src(), ids).numpy()
    print("tokens: llama.cpp %d, spiece %d" % (e.shape[0], len(ids)), flush=True)
    n = min(len(ids), e.shape[0])
    cos = (e[:n] * ref[:n]).sum(1) / (np.linalg.norm(e[:n], axis=1) * np.linalg.norm(ref[:n], axis=1))
    print("llama.cpp (city96 Q8_0) vs our Python encoder (our Q8_0): per-token cosine mean %.5f min %.5f" % (cos.mean(), cos.min()), flush=True)
