# 2026-10-08 night (goal: quantize umt5 to Q4_0): does the Q4_0 GGUF encode text like the fp8 original? A streaming
# umT5-XXL encoder in plain torch (one block's weights at a time, < 1 GB RAM): T5 RMSNorm, self-attention without the
# 1/sqrt(d) scale and with a relative-position bias per layer (umT5; 32 buckets, max distance 128, bidirectional),
# gated-GELU FFN (gelu_tanh(wi_0 x) * wi_1 x -> wo), final norm. The same prompt through the fp8 weights (with
# their scale_weight) and through the Q4_0 GGUF -> per-token cosine and relative L2 of the embeddings Wan cross-attends to.
#   python umt5_check.py ["prompt"]
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm
from safetensors import safe_open
import gguf

D, H, DK, NB, MAXD, NL = 4096, 64, 64, 32, 128, 24
try:
    import pulsex_paths as PX                                         # 10-24: every folder in one place (pulsex_paths.py)
except ImportError:                                                   # imported from outside wan22
    import os as _os
    import sys as _sys
    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    import pulsex_paths as PX

FP8 = str(PX.SRC / "umt5-xxl-fp8.safetensors")
Q4 = sys.argv[sys.argv.index("--q") + 1] if "--q" in sys.argv else str(PX.SRC / "umt5-xxl-encoder-Q4_0.gguf")   # the GGUF under test


class FP8Src:
    """the fp8 safetensors read with plain file reads by header offset (10-08: safetensors' memory-mapped torch tensors
    segfaulted in torch/storage.py beside the NPU video, while ReadFile of the same bytes was fine)"""
    DT = {"F8_E4M3": (np.uint8, "float8_e4m3fn"), "F16": (np.float16, None), "F32": (np.float32, None), "BF16": (np.uint16, "bfloat16")}

    def __init__(self):
        import json
        import struct
        with open(FP8, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h = json.loads(f.read(n))
        self.base = 8 + n
        self.keys = set(k for k in self.h if k != "__metadata__")

    def raw(self, k):
        h = self.h[k]
        npdt, tdt = self.DT[h["dtype"]]
        a, b = h["data_offsets"]
        with open(FP8, "rb") as f:
            f.seek(self.base + a)
            arr = np.frombuffer(f.read(b - a), dtype=npdt).reshape(h["shape"]).copy()
        t = torch.from_numpy(arr)
        return t.view(getattr(torch, tdt)) if tdt else t

    def get(self, k):
        t = self.raw(k)
        if t.dtype == getattr(torch, "float8_e4m3fn", None):
            base = k[:-len(".weight")]
            t = t.float() * self.raw(base + ".scale_weight").float()
        return t.float()


MIX = sys.argv[sys.argv.index("--mix") + 1] if "--mix" in sys.argv else None          # Q8_0 GGUF for the matched tensors
VARIANTS = sys.argv[sys.argv.index("--variants") + 1].split(";") if "--variants" in sys.argv else [""]   # regexes; "" = pure --q


def fp8_embedding_rows(ids):
    """rows of shared.weight (F16 in the fp8 file) read straight from the file by offset - no memory-mapped tensor"""
    import json
    import struct
    with open(FP8, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))["shared.weight"]
        assert h["dtype"] == "F16", h["dtype"]
        dim, base = h["shape"][1], 8 + n + h["data_offsets"][0]
        out = []
        for i in ids:
            f.seek(base + i * dim * 2)
            out.append(np.frombuffer(f.read(dim * 2), dtype=np.float16))
    return torch.from_numpy(np.stack(out).astype(np.float32))


def llama_to_hf(n):
    """llama.cpp T5-encoder tensor names (city96's umT5 GGUFs) -> the Hugging Face names this encoder reads"""
    import re
    if n == "token_embd.weight":
        return "shared.weight"
    if n == "enc.output_norm.weight":
        return "encoder.final_layer_norm.weight"
    m = re.match(r"enc\.blk\.(\d+)\.(\w+)\.weight$", n)
    if not m:
        return n
    i, k = m.groups()
    p = "encoder.block.%s.layer." % i
    return {"attn_norm": p + "0.layer_norm", "attn_q": p + "0.SelfAttention.q", "attn_k": p + "0.SelfAttention.k",
            "attn_v": p + "0.SelfAttention.v", "attn_o": p + "0.SelfAttention.o",
            "attn_rel_b": p + "0.SelfAttention.relative_attention_bias", "ffn_norm": p + "1.layer_norm",
            "ffn_gate": p + "1.DenseReluDense.wi_0", "ffn_up": p + "1.DenseReluDense.wi_1",
            "ffn_down": p + "1.DenseReluDense.wo"}.get(k, n) + ".weight"


class Q4Src:
    def __init__(self, pat=""):
        r = gguf.GGUFReader(Q4)
        self.t = {llama_to_hf(t.name): t for t in r.tensors}
        if pat:
            m = {t.name: t for t in gguf.GGUFReader(MIX).tensors}
            hit = [k for k in self.t if re.search(pat, k)]
            self.t.update({k: m[k] for k in hit})
            print("  variant %r: %d tensors from %s" % (pat, len(hit), Path(MIX).name), flush=True)

    def get(self, k):
        t = self.t[k]
        shape = tuple(int(s) for s in reversed(t.shape))
        if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            a = np.asarray(t.data, dtype=np.float32).reshape(shape)
        else:
            a = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape)
        return torch.from_numpy(np.ascontiguousarray(a))


def rel_bucket(rel):
    """T5 bidirectional relative-position buckets"""
    nb = NB // 2
    ret = (rel > 0).long() * nb
    n = rel.abs()
    max_exact = nb // 2
    small = n < max_exact
    large = max_exact + (torch.log(n.float().clamp(min=1) / max_exact) / math.log(MAXD / max_exact) * (nb - max_exact)).long()
    large = large.clamp(max=nb - 1)
    return ret + torch.where(small, n, large)


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def encode(src, ids):
    L = len(ids)
    x = None
    if not hasattr(src, "t"):                                          # fp8: only the rows of the prompt (10-08: the whole
        rows = fp8_embedding_rows(ids)                                  # 256k x 4096 table: torch's mmap view segfaulted
                                                                        # (10-08, beside the NPU video); plain file reads
        sk = "shared.scale_weight"
        x = rows.float() * (src.f.get_tensor(sk).float() if sk in src.keys else 1.0)
    if x is None:                                                      # the Q4 embedding: dequantize only the rows used
        t = src.t["shared.weight"]
        rows = gguf.quants.dequantize(t.data[np.array(ids)], t.tensor_type).astype(np.float32)
        x = torch.from_numpy(rows)
    pos = torch.arange(L)
    buckets = rel_bucket(pos[None, :] - pos[:, None])                    # [L, L]
    for i in range(NL):
        p = "encoder.block.%d.layer." % i
        h = rms(x, src.get(p + "0.layer_norm.weight"))
        q = (h @ src.get(p + "0.SelfAttention.q.weight").T).view(L, H, DK).transpose(0, 1)
        k = (h @ src.get(p + "0.SelfAttention.k.weight").T).view(L, H, DK).transpose(0, 1)
        v = (h @ src.get(p + "0.SelfAttention.v.weight").T).view(L, H, DK).transpose(0, 1)
        bias = src.get(p + "0.SelfAttention.relative_attention_bias.weight")[buckets].permute(2, 0, 1)   # [H, L, L]
        a = torch.softmax(q @ k.transpose(1, 2) + bias, -1) @ v
        x = x + a.transpose(0, 1).reshape(L, H * DK) @ src.get(p + "0.SelfAttention.o.weight").T
        h = rms(x, src.get(p + "1.layer_norm.weight"))
        g = torch.nn.functional.gelu(h @ src.get(p + "1.DenseReluDense.wi_0.weight").T, approximate="tanh")
        x = x + (g * (h @ src.get(p + "1.DenseReluDense.wi_1.weight").T)) @ src.get(p + "1.DenseReluDense.wo.weight").T
    return rms(x, src.get("encoder.final_layer_norm.weight"))


def main():
    torch.set_num_threads(6)
    prompt = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "A waterfall in a wide northern river, white foam swirling in the pool below the falls, spruce forest all around."
    sp = spm.SentencePieceProcessor(model_file=str(PX.SRC / "spiece.model"))
    ids = sp.encode(prompt) + [1]                                       # </s>
    print("tokens", len(ids), flush=True)
    with torch.no_grad():
        t0 = time.time()
        a = encode(FP8Src(), ids)
        print("fp8: %.1f s, out std %.4f, finite %s" % (time.time() - t0, float(a.std()), bool(torch.isfinite(a).all())), flush=True)
        for pat in VARIANTS:
            t0 = time.time()
            b = encode(Q4Src(pat), ids)
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            print("%s [%s] vs fp8: per-token cosine mean %.5f min %.5f, rel L2 %.3e (%.1f s)" % (Path(Q4).name, pat or "-", float(cos.mean()), float(cos.min()), float((a - b).norm() / a.norm()), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
