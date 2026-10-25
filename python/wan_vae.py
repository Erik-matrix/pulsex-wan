# 2026-10-07 night (goal: umT5 and the VAE in Q4_0, if it works). The Wan2.2 VAE
# (Wan2.2_VAE.safetensors in the source folder, 705 M params F16, original Wan key names) -> diffusers' AutoencoderKLWan
# (is_residual = the 2.2 variant) with our own key map (diffusers' single-file loader only knows the 2.1 layout), a Q4_0
# GGUF of it (conv weights flattened to [out, in*k] for the 32-element blocks, original shape in the metadata; tensors whose
# row is not a multiple of 32 stay F16), and the honest check: encode + decode a real picture with the F16 weights and
# with the Q4_0 weights, PSNR of each against the picture and against each other.
#   python wan_vae.py convert [Q4_0|Q8_0]      -> <source folder>\Wan2.2_VAE-<type>.gguf
#   python wan_vae.py check   [Q4_0|Q8_0]      -> PSNR table (CPU, low priority)
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import pulsex_paths as PX                                         # 10-24: every folder in one place (pulsex_paths.py)
except ImportError:                                                   # imported from outside wan22
    import os as _os
    import sys as _sys
    _sys.path.append(_os.path.dirname(_os.path.abspath(__file__)))
    import pulsex_paths as PX

DIR = PX.SRC                                                          # the source safetensors; convert() writes here (developer)
SRC = DIR / "Wan2.2_VAE.safetensors"
CFG = PX.model_file("vae_config", "config.json")                       # 10-24: a copy under MODELS, the source folder as the fallback


def map_key(k):
    """original Wan2.2 VAE name -> diffusers AutoencoderKLWan name"""
    if k.startswith("conv1."):
        return "quant_conv." + k[6:]
    if k.startswith("conv2."):
        return "post_quant_conv." + k[6:]
    part, rest = k.split(".", 1)                                       # encoder / decoder
    rest = re.sub(r"^conv1\.", "conv_in.", rest)
    rest = re.sub(r"^head\.0\.", "norm_out.", rest)
    rest = re.sub(r"^head\.2\.", "conv_out.", rest)
    rest = re.sub(r"^middle\.0\.", "mid_block.resnets.0.", rest)
    rest = re.sub(r"^middle\.1\.", "mid_block.attentions.0.", rest)
    rest = re.sub(r"^middle\.2\.", "mid_block.resnets.1.", rest)
    m = re.match(r"^(downsamples|upsamples)\.(\d+)\.(downsamples|upsamples)\.(\d+)\.(.*)$", rest)
    if m:
        kind, i, j, tail = m.group(1), int(m.group(2)), int(m.group(4)), m.group(5)
        nres = 2 if kind == "downsamples" else 3
        blk = "down_blocks" if kind == "downsamples" else "up_blocks"
        if j < nres:
            rest = "%s.%d.resnets.%d.%s" % (blk, i, j, tail)
        else:
            rest = "%s.%d.%s.%s" % (blk, i, "downsampler" if kind == "downsamples" else "upsampler", tail)
    rest = re.sub(r"residual\.0\.", "norm1.", rest)
    rest = re.sub(r"residual\.2\.", "conv1.", rest)
    rest = re.sub(r"residual\.3\.", "norm2.", rest)
    rest = re.sub(r"residual\.6\.", "conv2.", rest)
    rest = re.sub(r"(resnets\.\d+\.)shortcut\.", r"\1conv_shortcut.", rest)
    return part + "." + rest


def new_vae():
    from diffusers import AutoencoderKLWan
    cfg = json.loads(CFG.read_text())
    cfg.pop("_class_name", None)
    cfg.pop("clip_output", None)
    return AutoencoderKLWan(**cfg)


def load_f16():
    from safetensors.torch import load_file
    sd = {map_key(k): v.float() for k, v in load_file(str(SRC)).items()}
    vae = new_vae()
    ref = vae.state_dict()
    missing = [k for k in ref if k not in sd]
    extra = [k for k in sd if k not in ref]
    bad = [k for k in sd if k in ref and tuple(sd[k].shape) != tuple(ref[k].shape)]
    assert not missing and not extra and not bad, (missing[:5], extra[:5], bad[:5])
    vae.load_state_dict(sd)
    return vae.eval()


def convert(qname="Q4_0"):
    import gguf
    from safetensors.torch import load_file
    QT = getattr(gguf.GGMLQuantizationType, qname)
    out = DIR / ("Wan2.2_VAE-%s.gguf" % qname)
    w = gguf.GGUFWriter(str(out), "wan22vae")
    w.add_name("Wan2.2 VAE")
    nq = nf = 0
    for k, v in load_file(str(SRC)).items():
        a = v.float().numpy()
        name = map_key(k)
        flat = a.reshape(a.shape[0], -1) if a.ndim >= 2 else a
        if a.ndim >= 2 and flat.shape[1] % 32 == 0:
            q = gguf.quants.quantize(flat.astype(np.float32), QT)
            w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=QT)
            w.add_array("shape." + name, list(a.shape))
            nq += 1
        else:
            w.add_tensor(name, a.astype(np.float16))
            nf += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote %s: %d %s, %d F16, %.3f GB (source %.3f GB)" % (out, nq, qname, nf, out.stat().st_size / 1e9, SRC.stat().st_size / 1e9), flush=True)
    return out


def load_gguf(path):
    import gguf
    r = gguf.GGUFReader(str(path))
    shapes = {}
    for k, f in r.fields.items():
        if k.startswith("shape."):
            shapes[k[6:]] = [int(f.parts[i][0]) for i in f.data]
    sd = {}
    for t in r.tensors:
        if t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
            a = np.asarray(t.data, dtype=np.float32).reshape(tuple(int(s) for s in reversed(t.shape)))
        else:
            a = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        if t.name in shapes:
            a = a.reshape(shapes[t.name])
        sd[t.name] = torch.from_numpy(np.ascontiguousarray(a))
    vae = new_vae()
    vae.load_state_dict(sd)
    return vae.eval()


def load_gguf_decoder(path, lazy=False):
    """10-11 night (goal: the memory at the big VAE's start): load_gguf made every tensor f32 (encoder too) and
    then a whole randomly initialised model to copy them into - two full f32 VAEs at once, ~6 GB at the 720p VAE's
    start, for a CPU control that only decodes. Here: the model is built on the meta device (no memory), only
    post_quant_conv and the decoder are dequantised, and they are assigned in place (no second copy). The encoder and
    quant_conv stay meta: running them raises, decoding is the same numbers as load_gguf's (checked in the same run by
    tools/vae_loader_check_1011.py)."""
    import gguf
    r = gguf.GGUFReader(str(path))
    shapes = {}
    for k, f in r.fields.items():
        if k.startswith("shape."):
            shapes[k[6:]] = [int(f.parts[i][0]) for i in f.data]
    sd, later = {}, {}
    for t in r.tensors:
        if not (t.name.startswith("decoder.") or t.name.startswith("post_quant_conv.")):
            continue
        if t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
            a = np.asarray(t.data, dtype=np.float32).reshape(tuple(int(s) for s in reversed(t.shape)))
        elif lazy and t.name.startswith("decoder.") and t.name.endswith(".weight"):
            later[t.name] = t                                         # 10-12: stays Q4_0 in the memory-mapped file
            continue
        else:
            a = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        if t.name in shapes:
            a = a.reshape(shapes[t.name])
        sd[t.name] = torch.from_numpy(np.ascontiguousarray(a))
    with torch.device("meta"):
        vae = new_vae()
    missing, unexpected = vae.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected and all(k.startswith("encoder.") or k.startswith("quant_conv.") or k in later for k in missing), (unexpected, missing[:5])
    # 10-12 morning (lazy=True, the CPU control's memory): the decoder's Q4_0 weights (2.07 GB as f32) are dequantised
    # right before their module runs - the same gguf.quants.dequantize as above, so the same numbers - and dropped after
    for name, t in later.items():
        mod = vae.get_submodule(name[:-len(".weight")])
        shape = shapes.get(name) or tuple(int(s) for s in reversed(t.shape))

        def pre(m, args, t=t, shape=shape):
            m.weight = torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(
                gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape))), requires_grad=False)

        def post(m, args, out):
            m.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
        mod.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    vae._lazy_reader = r                                              # keeps the file mapped as long as the model lives
    leftover = [n for n, b in list(vae.named_parameters()) + list(vae.named_buffers())
                if b.is_meta and not (n.startswith("encoder.") or n.startswith("quant_conv."))]
    assert not leftover, leftover[:5]                                  # every decoding tensor is real (or comes when it runs)
    return vae.eval()


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse <= 1e-12 else 10 * math.log10(4.0 / mse)          # range -1..1


def check(qname="Q4_0"):
    from PIL import Image
    torch.set_num_threads(6)
    pic = Image.open(str(PX.WORK / "vae_check" / "picture.png")).convert("RGB").resize((832, 480))
    x = torch.from_numpy(np.asarray(pic, np.float32) / 127.5 - 1).permute(2, 0, 1)[None, :, None]   # [1, 3, 1, H, W]
    path = DIR / ("Wan2.2_VAE-%s.gguf" % qname)
    if not path.exists():
        convert(qname)
    out = {}
    for tag, loader in (("F16", load_f16), (qname, lambda: load_gguf(path))):
        t0 = time.time()
        vae = loader()
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode()
            y = vae.decode(z).sample.clamp(-1, 1)
        out[tag] = (z, y)
        print("%s: encode+decode %.1f s, latent %s, PSNR vs picture %.2f dB" % (tag, time.time() - t0, list(z.shape), psnr(y, x)), flush=True)
    zf, yf = out["F16"]
    zq, yq = out[qname]
    print("%s vs F16: decoded PSNR %.2f dB, latent rel L2 %.3e" % (qname, psnr(yq, yf), float((zq - zf).norm() / zf.norm())), flush=True)
    for tag, (_, y) in out.items():
        Image.fromarray(((y[0, :, 0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)).save(
            str(PX.WORK / "vae_check" / ("vae_roundtrip_%s.png" % tag)))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    q = sys.argv[2] if len(sys.argv) > 2 else "Q4_0"
    convert(q) if cmd == "convert" else check(q)
