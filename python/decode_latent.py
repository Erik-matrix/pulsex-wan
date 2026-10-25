# 2026-10-08: the first NPU video (480x832, 49 frames, 25 steps) finished all steps and saved latent.npy at 10:51,
# then the process ended without frames somewhere in the VAE decode. This decodes a saved latent on its own (CPU, low
# priority), prints the peak memory and the traceback if it fails, writes the PNG frames and the mp4.
#   python decode_latent.py [latent.npy] [--tag wan_480x832_f49_s25_seed1] [--threads 5]
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wan_vae   # noqa: E402
import pulsex_paths as PX   # noqa: E402  (10-24: every folder in one place)



def conv3d_as_conv2d():
    """10-08: torch's conv3d ran at ~1.7 of 6 cores on the ARM CPU; out[t] = sum_k conv2d(x[t + k], w[:, :, k]) gives the
    same numbers (1.2e-5) 1.9x faster at 2 threads (conv3d_probe.py). WanCausalConv3d pads explicitly before its conv3d,
    so only that inner call is replaced, and only for stride 1 / dilation 1 / groups 1."""
    import torch.nn.functional as F
    from diffusers.models.autoencoders import autoencoder_kl_wan as m
    orig = m.WanCausalConv3d.forward

    def conv(self, x):
        w = self.weight
        if tuple(self.stride) != (1, 1, 1) or tuple(self.dilation) != (1, 1, 1) or self.groups != 1 or x.shape[0] != 1:
            return F.conv3d(x, w, self.bias, self.stride, 0, self.dilation, self.groups)
        kT = w.shape[2]
        Tout = x.shape[2] - kT + 1
        frames = x[0].transpose(0, 1)                                    # [T, C, H, W]: the frames as the batch
        out = None
        for k in range(kT):
            y = F.conv2d(frames[k:k + Tout], w[:, :, k], None)
            out = y if out is None else out + y
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out.transpose(0, 1).unsqueeze(0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x.to(x.device), x], dim=2)
            padding[4] -= cache_x.shape[2]
        return conv(self, F.pad(x, padding))

    # known-healthy control in the same run: one real-shaped causal conv both ways, with a cache
    c = m.WanCausalConv3d(32, 32, 3, padding=1)
    xi, ca = torch.randn(1, 32, 4, 24, 24), torch.randn(1, 32, 2, 24, 24)
    with torch.no_grad():
        a, b = orig(c, xi, ca), forward(c, xi, ca)
    d = float((a - b).abs().max())
    print("conv3d -> conv2d control: max |diff| %.2e" % d, flush=True)
    assert d < 1e-4, d
    m.WanCausalConv3d.forward = forward


def peak_gb():
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / 1e9
    except Exception:
        return float("nan")


def main():
    global VAE
    src = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else PX.OUT / "latent.npy"
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "wan_480x832_f49_s25_seed1"
    torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 5)
    out = src.parent
    VAE = PX.model_file("Wan2.2_VAE-Q4_0.gguf")                    # MODELS, else the source folder (10-24)
    # (the old "C:, else D:" fallback is PX.model_file's)
    t0 = time.time()
    try:
        lat = torch.from_numpy(np.load(src))
        print("latent %s, finite %s, std %.3f" % (list(lat.shape), bool(torch.isfinite(lat).all()), float(lat.std())), flush=True)
        cfg = json.loads(wan_vae.CFG.read_text())
        mean = torch.tensor(cfg["latents_mean"]).view(1, 48, 1, 1, 1)
        std = torch.tensor(cfg["latents_std"]).view(1, 48, 1, 1, 1)
        if "--conv3d" not in sys.argv:
            conv3d_as_conv2d()
        vae = wan_vae.load_gguf(VAE)
        # whole-frame decode asked conv3d for 22 GB at 480x832 x 4 frames (the 1-frame CPU picture fitted): spatial tiles
        # with blended overlap, diffusers' own tiled_decode; 256 px tiles / 192 px stride = 15 tiles, ~1/6 of the allocation
        tile = int(sys.argv[sys.argv.index("--tile") + 1]) if "--tile" in sys.argv else 256
        vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                          tile_sample_stride_height=tile * 3 // 4, tile_sample_stride_width=tile * 3 // 4)
        print("VAE %s loaded %.1f s, peak %.2f GB" % (VAE, time.time() - t0, peak_gb()), flush=True)
        with torch.no_grad():
            y = vae.decode(lat[None] * std + mean).sample.clamp(-1, 1)[0]   # [3, T, H, W]
        print("decoded %s in %.1f s, peak %.2f GB" % (list(y.shape), time.time() - t0, peak_gb()), flush=True)
        from PIL import Image
        for k in range(y.shape[1]):
            Image.fromarray(((y[:, k].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)).save(out / ("%s_%03d.png" % (tag, k)))
        mp4 = out / "decoded.mp4"
        subprocess.run([str(PX.FFMPEG), "-y", "-hide_banner", "-loglevel", "error", "-framerate", "24",
                        "-i", str(out / (tag + "_%03d.png")), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(mp4)], check=True)
        print("wrote %d frames + %s (%.1f MB), total %.1f s" % (y.shape[1], mp4, mp4.stat().st_size / 1e6, time.time() - t0), flush=True)
    except BaseException:
        print("FAILED after %.1f s, peak %.2f GB" % (time.time() - t0, peak_gb()), flush=True)
        traceback.print_exc()
        sys.exit(1)



if __name__ == "__main__":
    main()
