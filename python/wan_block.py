# 2026-10-07 evening: another way in - let Qualcomm's tools read the whole model instead of hand-building
# every graph and mask. Spike step 1-2: one Wan2.2-TI2V-5B DiT block in PyTorch, weights dequantized from the GGUF
# (Wan2.2-TI2V-5B-Q4_0.gguf, the full DiT file), written to export cleanly to ONNX for QAIRT / AI Hub:
#  * RMSNorm max-scaled (QNN squares in fp16 and saturates at 2^17 - LTX VAE, 09-29),
#  * GELU written out with tanh (QNN's Gelu is GELU_QUICK - LTX engine, 09-28),
#  * RoPE with real cos/sin tables (rank <= 5), attention optionally in query chunks (the L x L scores are huge).
# Wan block (WanAttentionBlock): e = modulation + e0 (6 x dim);
#   x += self_attn(LN(x) (1 + e1) + e0) * e2;  x += cross_attn(LN3(x), ctx);  x += ffn(LN(x) (1 + e4) + e3) * e5
import math
import re

import numpy as np
import torch
import torch.nn as nn

try:
    import pulsex_paths as PX                                         # 10-24: every folder in one place (pulsex_paths.py)
except ImportError:                                                   # imported from outside wan22
    import os as _os
    import sys as _sys
    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    import pulsex_paths as PX

GGUF = str(PX.SRC / "Wan2.2-TI2V-5B-Q4_0.gguf")                      # the full DiT GGUF: --backend cpu / --demo-cpu only
DIM, FFN, HEADS = 3072, 14336, 24
HD = DIM // HEADS                                                     # 128


def load_tensors(prefix, path=GGUF):
    """the GGUF tensors whose names start with prefix, dequantized to float32 numpy in torch's (out, in) order"""
    import gguf
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        if not t.name.startswith(prefix):
            continue
        shape = tuple(int(s) for s in reversed(t.shape))
        if t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
            a = np.asarray(t.data, dtype=np.float32).reshape(shape)
        else:
            a = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape)
        out[t.name[len(prefix):]] = a
    return out


def rms_norm(x, w, eps=1e-6):
    """max-scaled RMSNorm: exact, but the squares stay small in fp16"""
    m = x.abs().amax(-1, keepdim=True).clamp(min=1e-6)
    y = x / m
    return y * torch.rsqrt((y * y).mean(-1, keepdim=True) + eps / (m * m)) * w


def gelu_tanh(x):
    """GELU, tanh approximation. 10-09: written as x * sigmoid(2u) (== 0.5 x (1 + tanh u)) - QNN's Tanh was 44 % of
    part B on the HTP (AI Hub profile); WAN_GELU=tanh keeps the old graph form"""
    import os
    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    if os.environ.get("WAN_GELU", "tanh") == "tanh":                   # 10-09: sigmoid form measured 101.7 vs 102.0 ms - no gain, the compiled graphs use tanh
        return 0.5 * x * (1 + torch.tanh(u))
    return x * torch.sigmoid(2 * u)


def rope_tables(F, Hh, Ww, theta=10000.0):
    """Wan's 3D RoPE split of the 64 complex pairs per head: frames 22, height 21, width 21 -> cos, sin [L, 64]"""
    d = HD
    dims = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]          # 44, 42, 42
    fr = [1.0 / theta ** (torch.arange(0, k, 2, dtype=torch.float64) / k) for k in dims]
    f = torch.arange(F, dtype=torch.float64)[:, None, None, None] * fr[0][None, None, None, :]
    h = torch.arange(Hh, dtype=torch.float64)[None, :, None, None] * fr[1][None, None, None, :]
    w = torch.arange(Ww, dtype=torch.float64)[None, None, :, None] * fr[2][None, None, None, :]
    ang = torch.cat([f.expand(F, Hh, Ww, -1), h.expand(F, Hh, Ww, -1), w.expand(F, Hh, Ww, -1)], -1).reshape(F * Hh * Ww, d // 2)
    return torch.cos(ang).float(), torch.sin(ang).float()


def apply_rope(x, cos, sin):
    """x [1, L, heads, 128] -> rotated pairs (x0, x1) as complex multiply by e^(i ang); rank <= 5"""
    L = x.shape[1]
    xp = x.reshape(1, L, HEADS, HD // 2, 2)
    x0, x1 = xp[..., 0], xp[..., 1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), -1).reshape(1, L, HEADS, HD)


def attention(q, k, v, chunk=0):
    """q [1, Lq, H, D], k/v [1, Lk, H, D] -> [1, Lq, H*D]; queries in chunks when chunk > 0"""
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)    # [1, H, L, D]
    scale = 1.0 / math.sqrt(HD)
    outs = []
    Lq = q.shape[2]
    step = chunk if chunk > 0 else Lq
    for i in range(0, Lq, step):
        s = torch.matmul(q[:, :, i:i + step] * scale, k.transpose(-1, -2))
        outs.append(torch.matmul(torch.softmax(s, -1), v))
    o = torch.cat(outs, 2) if len(outs) > 1 else outs[0]
    return o.transpose(1, 2).reshape(1, Lq, DIM)


class WanBlock(nn.Module):
    def __init__(self, w, attn_chunk=0):
        super().__init__()
        self.attn_chunk = attn_chunk
        P = lambda a: nn.Parameter(torch.from_numpy(np.ascontiguousarray(a)), requires_grad=False)
        self.w = nn.ParameterDict({k.replace(".", "_"): P(v) for k, v in w.items()})

    def lin(self, x, name):
        return torch.nn.functional.linear(x, self.w[name.replace(".", "_") + "_weight"], self.w[name.replace(".", "_") + "_bias"])

    def forward(self, x, e0, ctx, cos, sin):
        """x [1, L, 3072], e0 [1, 6, 3072] (the time projection), ctx [1, Lc, 3072], cos/sin [L, 64].
        e0 [1, L, 6, 3072]: per token (10-10, image to video: the first frame's tokens at t = 0)"""
        if e0.dim() == 4:
            e = self.w["modulation"].reshape(1, 1, 6, DIM) + e0
            e_0, e_1, e_2, e_3, e_4, e_5 = (e[:, :, k] for k in range(6))   # [1, L, 3072] each
        else:
            e = self.w["modulation"].reshape(1, 6, DIM) + e0
            e_0, e_1, e_2, e_3, e_4, e_5 = e.split(1, dim=1)            # ONNX Split, not Slice (QAIRT 2.45 dropped a Slice's start feeding an Add)
        ln = lambda t: torch.nn.functional.layer_norm(t, (DIM,), eps=1e-6)
        L = x.shape[1]
        # self attention
        h = ln(x) * (1 + e_1) + e_0
        q = rms_norm(self.lin(h, "self_attn.q"), self.w["self_attn_norm_q_weight"]).reshape(1, L, HEADS, HD)
        k = rms_norm(self.lin(h, "self_attn.k"), self.w["self_attn_norm_k_weight"]).reshape(1, L, HEADS, HD)
        v = self.lin(h, "self_attn.v").reshape(1, L, HEADS, HD)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        x = x + self.lin(attention(q, k, v, self.attn_chunk), "self_attn.o") * e_2
        # cross attention (text)
        h = torch.nn.functional.layer_norm(x, (DIM,), self.w["norm3_weight"], self.w["norm3_bias"], eps=1e-6)
        Lc = ctx.shape[1]
        q = rms_norm(self.lin(h, "cross_attn.q"), self.w["cross_attn_norm_q_weight"]).reshape(1, L, HEADS, HD)
        k = rms_norm(self.lin(ctx, "cross_attn.k"), self.w["cross_attn_norm_k_weight"]).reshape(1, Lc, HEADS, HD)
        v = self.lin(ctx, "cross_attn.v").reshape(1, Lc, HEADS, HD)
        x = x + self.lin(attention(q, k, v, self.attn_chunk), "cross_attn.o")
        # feed forward
        h = ln(x) * (1 + e_4) + e_3
        x = x + self.lin(gelu_tanh(self.lin(h, "ffn.0")), "ffn.2") * e_5
        return x


class WanBlockA(nn.Module):
    """part A for QNN: modulated LN + q/k/v + RMSNorm + RoPE -> q, k, v [1, L, 3072] (self-attention runs outside:
    our own HMX flash attention in VTCM tiles - no L x L matrix in a QNN graph)"""
    def __init__(self, blk):
        super().__init__()
        self.b = blk

    def forward(self, x, shift, scale1, cos, sin):
        """shift = e_0, scale1 = 1 + e_1 [1, 1, 3072], computed on the CPU per step (no Split/Slice in the graph:
        the converter mis-shaped the modulation split, 'Unable to broadcast [1,5070,3072] and [1,3,3072]')"""
        b = self.b
        L = x.shape[1]
        h = torch.nn.functional.layer_norm(x, (DIM,), eps=1e-6) * scale1 + shift
        q = rms_norm(b.lin(h, "self_attn.q"), b.w["self_attn_norm_q_weight"]).reshape(1, L, HEADS, HD)
        k = rms_norm(b.lin(h, "self_attn.k"), b.w["self_attn_norm_k_weight"]).reshape(1, L, HEADS, HD)
        v = b.lin(h, "self_attn.v")
        return apply_rope(q, cos, sin).reshape(1, L, DIM), apply_rope(k, cos, sin).reshape(1, L, DIM), v


class WanBlockB(nn.Module):
    """part B for QNN: x + o(attn) e2, cross attention to the text (Lc = 512 keys: small scores), modulated FFN"""
    def __init__(self, blk):
        super().__init__()
        self.b = blk

    def forward(self, x, attn, gate1, shift2, scale2, gate2, ctx):
        """gate1 = e_2, shift2 = e_3, scale2 = 1 + e_4, gate2 = e_5 [1, 1, 3072], computed on the CPU per step"""
        b = self.b
        L, Lc = x.shape[1], ctx.shape[1]
        x = x + b.lin(attn, "self_attn.o") * gate1
        h = torch.nn.functional.layer_norm(x, (DIM,), b.w["norm3_weight"], b.w["norm3_bias"], eps=1e-6)
        q = rms_norm(b.lin(h, "cross_attn.q"), b.w["cross_attn_norm_q_weight"]).reshape(1, L, HEADS, HD)
        k = rms_norm(b.lin(ctx, "cross_attn.k"), b.w["cross_attn_norm_k_weight"]).reshape(1, Lc, HEADS, HD)
        v = b.lin(ctx, "cross_attn.v").reshape(1, Lc, HEADS, HD)
        x = x + b.lin(attention(q, k, v, 0), "cross_attn.o")
        h = torch.nn.functional.layer_norm(x, (DIM,), eps=1e-6) * scale2 + shift2
        return x + b.lin(gelu_tanh(b.lin(h, "ffn.0")), "ffn.2") * gate2


def cross_kv(blk, ctx):
    """the cross-attention k (RMS-normed) and v of the text, [1, Lc, 3072] each - constant for a prompt, so computed once
    per block and prompt (10-09) instead of in every tile of every step and CFG pass"""
    k = rms_norm(blk.lin(ctx, "cross_attn.k"), blk.w["cross_attn_norm_k_weight"])
    return k, blk.lin(ctx, "cross_attn.v")


class WanBlockB3(nn.Module):
    """part B with the text's cross-attention k/v as inputs (cross_kv) instead of ctx: no 2 x 512 x 3072^2 MAC per tile"""
    def __init__(self, blk):
        super().__init__()
        self.b = blk

    def forward(self, x, attn, gate1, shift2, scale2, gate2, kc, vc):
        b = self.b
        L, Lc = x.shape[1], kc.shape[1]
        x = x + b.lin(attn, "self_attn.o") * gate1
        h = torch.nn.functional.layer_norm(x, (DIM,), b.w["norm3_weight"], b.w["norm3_bias"], eps=1e-6)
        q = rms_norm(b.lin(h, "cross_attn.q"), b.w["cross_attn_norm_q_weight"]).reshape(1, L, HEADS, HD)
        x = x + b.lin(attention(q, kc.reshape(1, Lc, HEADS, HD), vc.reshape(1, Lc, HEADS, HD), 0), "cross_attn.o")
        h = torch.nn.functional.layer_norm(x, (DIM,), eps=1e-6) * scale2 + shift2
        return x + b.lin(gelu_tanh(b.lin(h, "ffn.0")), "ffn.2") * gate2


def half_split_perm():
    """per head, channel 2i -> i and 2i+1 -> 64+i: the RoPE pairs become two contiguous halves (no Gather on HTP)"""
    idx = []
    for h in range(HEADS):
        base = h * HD
        idx += [base + 2 * i for i in range(HD // 2)] + [base + 2 * i + 1 for i in range(HD // 2)]
    return torch.tensor(idx)


class WanBlockA3(nn.Module):
    """part A with the q/k weights permuted to the half-split layout: RoPE = (a, b) -> (a c - b s, a s + b c) on two
    contiguous halves. q and k come out permuted the same way per head - q.k (the attention) is unchanged; v as is."""
    def __init__(self, blk):
        super().__init__()
        pm = half_split_perm()
        P = lambda t: nn.Parameter(t.detach().clone(), requires_grad=False)
        w = blk.w
        self.wq, self.bq, self.nq = P(w["self_attn_q_weight"][pm]), P(w["self_attn_q_bias"][pm]), P(w["self_attn_norm_q_weight"][pm])
        self.wk, self.bk, self.nk = P(w["self_attn_k_weight"][pm]), P(w["self_attn_k_bias"][pm]), P(w["self_attn_norm_k_weight"][pm])
        self.wv, self.bv = P(w["self_attn_v_weight"]), P(w["self_attn_v_bias"])

    def rope(self, x, cos, sin):
        L = x.shape[1]
        a, b = x.reshape(1, L, HEADS, HD).split(HD // 2, dim=-1)
        c, s = cos[None, :, None, :], sin[None, :, None, :]
        return torch.cat((a * c - b * s, a * s + b * c), -1).reshape(1, L, DIM)

    def forward(self, x, shift, scale1, cos, sin):
        F_ = torch.nn.functional
        h = F_.layer_norm(x, (DIM,), eps=1e-6) * scale1 + shift
        q = rms_norm(F_.linear(h, self.wq, self.bq), self.nq)
        k = rms_norm(F_.linear(h, self.wk, self.bk), self.nk)
        v = F_.linear(h, self.wv, self.bv)
        return self.rope(q, cos, sin), self.rope(k, cos, sin), v


class WanBlockA4(WanBlockA3):
    """A3 with RMSNorm(x) w written as L2Norm(x) (sqrt(D) w): QNN has L2Norm as its own op; the hand-written max-scaled
    norm's y*y took 298 ms per tensor on the NPU (two of A3's 905 ms)"""
    def __init__(self, blk):
        super().__init__(blk)
        self.nq.data = self.nq.data * math.sqrt(DIM)
        self.nk.data = self.nk.data * math.sqrt(DIM)

    def forward(self, x, shift, scale1, cos, sin):
        F_ = torch.nn.functional
        h = F_.layer_norm(x, (DIM,), eps=1e-6) * scale1 + shift
        q = F_.normalize(F_.linear(h, self.wq, self.bq), p=2.0, dim=-1, eps=1e-12) * self.nq
        k = F_.normalize(F_.linear(h, self.wk, self.bk), p=2.0, dim=-1, eps=1e-12) * self.nk
        v = F_.linear(h, self.wv, self.bv)
        return self.rope(q, cos, sin), self.rope(k, cos, sin), v


def modulation_inputs(blk, e0):
    """the six modulation vectors of a block for one step, [1, 1, 3072] each: shift1, scale1, gate1, shift2, scale2, gate2"""
    e = blk.w["modulation"].reshape(1, 6, DIM) + e0
    return e[:, 0:1], 1 + e[:, 1:2], e[:, 2:3], e[:, 3:4], 1 + e[:, 4:5], e[:, 5:6]


def load_block(i, attn_chunk=0):
    w = load_tensors("blocks.%d." % i)
    return WanBlock(w, attn_chunk).eval()


if __name__ == "__main__":
    import sys
    import time
    F, Hh, Ww, Lc = 2, 15, 26, 512                                  # small: 2 latent frames at 480x832
    t0 = time.time()
    blk = load_block(0)
    print("block 0 loaded %.1f s, %d tensors" % (time.time() - t0, len(blk.w)))
    L = F * Hh * Ww
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, L, DIM, generator=g)
    e0 = 0.05 * torch.randn(1, 6, DIM, generator=g)
    ctx = torch.randn(1, Lc, DIM, generator=g)
    cos, sin = rope_tables(F, Hh, Ww)
    with torch.no_grad():
        t0 = time.time()
        y = blk(x, e0, ctx, cos, sin)
        print("forward L=%d: %.2f s, out mean %.4f std %.4f, finite %s" % (L, time.time() - t0, float(y.mean()), float(y.std()), bool(torch.isfinite(y).all())))
        blk.attn_chunk = 256
        yc = blk(x, e0, ctx, cos, sin)
        print("chunked attention max |diff| %.2e" % float((yc - y).abs().max()))
