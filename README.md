# PulseX Wan

Short videos from a description or from a picture, made on the NPU of a Snapdragon X laptop: no graphics card, no
cloud. It runs **Wan 2.2 TI2V-5B** (Wan-AI), its transformer quantised to 4 bits, on the Hexagon NPU, with the
VAE decoder as compiled NPU graphs.

## Examples

Made on a Snapdragon X Plus (16 GB) with the window's defaults: 15 steps, the *Sharp* decoder, 2 s (49 frames). The
pictures below are animated WebPs; each links to its mp4.

| Text to video, 480p | Text to video, 720p |
|---|---|
| [![a red boat on a misty lake](docs/examples/boat-480p.webp)](docs/examples/boat-480p.mp4) | [![a lighthouse at dusk](docs/examples/lighthouse-720p.webp)](docs/examples/lighthouse-720p.mp4) |
| *"a small red wooden boat drifting on a calm lake at sunrise, thin mist over the water, gentle ripples, a pine forest on the far shore"* - seed 4242, 5.5 min | *"a lighthouse on a rocky coast at dusk, waves breaking slowly on the rocks, seagulls gliding over the sea, static camera"* - seed 7, 12.5 min |

| Start picture | Image to video, 720p, *Calm* |
|---|---|
| [![a windmill on a hill](docs/examples/windmill-start.jpg)](docs/examples/windmill-start.jpg) | [![the windmill's sails turning](docs/examples/windmill-720p.webp)](docs/examples/windmill-720p.mp4) |
| made on the same NPU with [PulseX Image](https://github.com/Erik-matrix/pulsex-image) (FLUX.2 klein): *"an old wooden windmill on a green hill under a blue sky with white clouds, wildflowers in the grass, summer, photo"* | *"the windmill's sails turn slowly, white clouds drift across the blue sky, the grass and the wildflowers sway in the wind, static camera"* - seed 3, 12.5 min |

## What it does

- **Text to video**: a description in, an mp4 out, 2 s (49 frames) or 5 s (121 frames) at 24 frames a second.
- **Image to video**: start from a picture of your own (drop it on the window); it becomes the first frame and the
  description says what moves. *Calm* or *Lively* motion.
- **Two sizes**: 832 x 480 (480p, the fastest) and 1280 x 704 (720p).
- **Two decoders**: *Sharp* - the full Wan 2.2 VAE on the NPU, the sharpest picture; *Fast* - TAEHV, a tiny decoder,
  much faster and a little softer.
- Videos are saved in `Videos\PulseX Wan`, never over an existing file, and the window remembers the description,
  seed and settings behind each one. The newest video plays in the window.

## Measured

Snapdragon X Plus X1P-42-100 (Hexagon NPU v73), 16 GB, the whole chain (text encoder, transformer, VAE, mp4):

| | Time |
|---|---|
| Text to video, 720p, 5 s (121 frames, 15 steps) | about 30 min |
| Image to video, 480p, 2 s (49 frames, 15 steps) | about 7.3 min |

**Memory**: a 720p 5 s video adds about **5 GB** to the computer's committed memory at its peak; measured on a 16 GB
laptop (about 15.6 GB usable), the total peaked at about **13.9 GB** with 9 GB already in use by other programs.
Close what you can while a video is made; with less than 16 GB the long 720p videos will not fit.

The computer can be used meanwhile (the pipeline runs at below-normal priority and leaves two cores free), but a
video cannot be stopped halfway: an engine is never killed with the NPU open.

## What you need

- A **Snapdragon X** laptop with Windows 11 on ARM64 and **16 GB** of memory. Tested on Snapdragon X Plus
  (X1P-42-100, Hexagon NPU v73). Other Snapdragon X chips are untested; the VAE graphs on Hugging Face are compiled
  for this chip.
- The [Microsoft Visual C++ Redistributable for ARM64](https://aka.ms/vs/17/release/vc_redist.arm64.exe).
- **Python 3.12 for ARM64** (the "Windows installer (ARM64)" from python.org) and the packages:

  ```
  python -m pip install -r requirements.txt
  ```

  (about 0.6 GB; PyTorch comes from PyTorch's own index, which the file names.)
- Five files of **Qualcomm's AI runtime (QAIRT), all from one and the same SDK build**: `QnnHtp.dll`,
  `QnnSystem.dll` and `QnnHtpV73Stub.dll` from its `lib\aarch64-windows-msvc`, and the NPU-side module
  `libQnnHtpV73Skel.so` with its catalog `libqnnhtpv73.cat` from its `lib\hexagon-v73\unsigned`. PulseX Wan is
  tested with QAIRT **2.45.0** (build 2.45.0.260326154327, the build its QNN graphs were compiled with); the setup
  file ships exactly that set. `QnnHtp.dll` accepts only a skel of its own build: with a mixed set it falls back,
  without an error, to the NPU driver's own files, and the log says so in a `WARNING: QNN (...)` line. They are
  Qualcomm's and are not in this repository. (The copies inside the `onnxruntime-qnn` Python package are older and
  refuse the VAE graphs.)
- **ffmpeg** for writing the mp4: a Windows ARM64 build of `ffmpeg.exe` with libx264. The setup file downloads one
  fixed build, `ffmpeg-N-125875-g5d4d3bdc61-winarm64-gpl.zip` from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases/tag/autobuild-2026-07-31-14-10) (release `autobuild-2026-07-31-14-10`),
  checks it against its SHA-256 `b254b9248d9247d3c87f59ae6ec29fe12fe3644569ed9e04ca8265625e6f2fc9`, and puts its `ffmpeg.exe` in `bin\`
  with the build's license as `bin\LICENSE-ffmpeg.txt`. That build is FFmpeg under the **GNU GPL version 3** (it
  includes libx264); it is not part of PulseX and not covered by its MIT license. By hand: put any such
  `ffmpeg.exe` in `bin\` or on the PATH, or point `PULSEX_FFMPEG` at it.
- The **models** from Hugging Face and two original repositories (below), about 12.7 GB, on a fast SSD (an SD card
  makes loading many times slower).

> **The NPU module is self-signed.** Windows only loads `bin\libggml-htp-v73.so` on a machine set up to accept
> self-signed NPU modules (a developer setup). On a stock machine the NPU part is refused; that needs a
> Microsoft-attestation-signed module, which this project does not have yet.

## Install

The easy way: download `PulseX-Wan-Setup-0.1.0.exe` from the
[latest release](https://github.com/Erik-matrix/pulsex-wan/releases/latest) and run it. It installs the program
for your user only (no administrator), with the five Qualcomm files in `qnn\`, and downloads the models (about
12.7 GB, from Hugging Face, each file checked against its SHA-256) and ffmpeg (above). It does not install Python:
when it finds Python 3.12 for ARM64 it offers to run `python -m pip install -r requirements.txt` for you (about
0.6 GB, in a window of its own); otherwise install Python 3.12 ARM64 from
<https://www.python.org/downloads/windows/> and then run *Install the Python packages* from the Start menu. The
install folder must have a plain A-Z path; the setup suggests one. Uninstalling removes the program, the downloaded
models and the work files (with the text-encoder cache), never your videos.

By hand, without the setup file: the steps below.

## Set up

1. Take the zip of the [latest release](https://github.com/Erik-matrix/pulsex-wan/releases/latest) or clone the
   repository, into a folder whose path is plain A-Z (for example `C:\PulseX\wan`): the NPU driver only loads its
   runtime from such a folder.
2. Install Python 3.12 ARM64 and `python -m pip install -r requirements.txt`.
3. Put the five Qualcomm files in a folder `qnn` next to `bin`.
4. Get the models into a folder `models` next to `bin`:

   ```
   hf download Pexqman/Wan2.2-TI2V-5B-hexpack --local-dir models
   hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models
   hf download Wan-AI/Wan2.2-TI2V-5B google/umt5-xxl/spiece.model --local-dir models
   ```

   then rename `umt5-xxl-encoder-Q8_0.gguf` to `umt5-xxl-encoder-Q8_0-city96.gguf`, move
   `models\google\umt5-xxl\spiece.model` to `models\spiece.model`, and copy `third_party\taehv\taehv.py` into
   `models\taehv\` (the TAEHV weights and their license come with the first download).
   (`pip install huggingface_hub` gives the `hf` command.) The model card on Hugging Face lists every file with its
   SHA-256.
5. Run `bin\pulsex-wan.exe`.

```
pulsex-wan\
  bin\          the programs (this repository) + ffmpeg.exe if you put it here
  python\       the pipeline (wan_pipeline.py and its modules)
  third_party\  taehv
  qnn\          QnnHtp.dll, QnnSystem.dll, QnnHtpV73Stub.dll, libQnnHtpV73Skel.so, libqnnhtpv73.cat
  models\
    Wan2.2-TI2V-5B-Q4_0-ffn2q8-ggml-slim.gguf + .hexpack     the transformer (slim metadata + the NPU's own layout)
    Wan2.2-TI2V-5B-npu-extras.gguf                            text / time embeddings, patch and head
    umt5-xxl-encoder-Q8_0 GGUF, spiece.model                  the text encoder and its tokenizer
    Wan2.2_VAE-Q4_0.gguf + its caches, vae_config\            the VAE
    vae_npu_first_p3\, vae_npu_step_w8a8\                     the Sharp decoder, compiled for the NPU
    vae_npu_enc\, vae_npu_enc_704x1280\                       the picture encoder (image to video, 480p / 720p)
    taew_npu_ctx\, taehv\taew2_2_super.pth                    the Fast decoder
```

`bin\wan.json` says where these are (paths relative to itself); edit it to keep the models elsewhere, or set the
environment variables `PULSEX_WAN_MODELS`, `PULSEX_WAN_BIN`, `PULSEX_QNN`, `PULSEX_FFMPEG`, `PULSEX_WAN_DATA` (work
files, the window's settings and its list of videos; by default `data\` in the install folder) and `PULSEX_WAN_OUT` (the videos). `PULSEX_PYTHON` names the Python to use if it is not the standard
per-user Python 3.12 ARM64. The window offers what it finds: *Sharp* needs its decoder graphs, *Fast* the TAEHV
files, a start picture the encoder of that size.

## Use

Type a description (English works best: say what moves and how the camera stands), choose the length (2 s / 5 s),
the size (480p / 720p), the decoder (Sharp / Fast) and the steps (15, 12 or 10), and press **Make the film**.
For image to video, choose a picture (or drop it on the window); it is cropped to the chosen size, and *Calm* /
*Lively* sets how much moves. The bar shows the stage and about how long is left. Under the video: play it, open its
folder, use its description again; *Earlier films* lists the ones made before.

Closing the window while a video is being made waits for the video.

## The command line

The window runs the Python pipeline, which you can use directly (from the `python` folder):

```
python wan_pipeline.py "a red fox walking through a snowy forest at dawn, static camera" --backend ggml
       [--size 480x832|704x1280] [--frames 49|121] [--steps 15] [--seed N] [--solver unipc] [--cfg-until 0.4]
       [--vae npu_full|taew|taew_cpu] [--image picture.png --guide 3] [--name NAME --outdir FOLDER]
```

`--vae npu_full` is *Sharp*, `taew` is *Fast* on the NPU (480p, 49 frames), `taew_cpu` is *Fast* on the CPU (any
size and length).

## How it works

| Stage | Program | Where |
|---|---|---|
| text | `llama-embedding.exe` (umT5-XXL encoder, Q8_0, through ggml) | CPU |
| motion | `wan-stream.exe`: the 30 transformer blocks, 15 steps (UniPC) with guidance for the first 40 % | NPU (ggml-hexagon) |
| patch, head, solver | `wan_pipeline.py` (PyTorch) | CPU |
| picture in | the Wan 2.2 VAE encoder as a compiled QNN graph | NPU (ONNX Runtime, QNN EP) |
| frames | *Sharp*: the Wan 2.2 VAE decoder as compiled QNN graphs, in tiles, step by step through the frames; *Fast*: TAEHV | NPU (*Fast* at 720p or 5 s: CPU) |
| mp4 | ffmpeg | CPU |

The transformer's weights never sit in memory all at once: a block is read from the `.hexpack` (the NPU's own
layout, written once), used, and replaced by the next, with the following block read ahead. That is why a 5-billion
parameter video model runs on a 16 GB laptop, with 27 000 tokens of attention for a 720p 5 s video. A "slim" GGUF holds the metadata and the
few tensors the processor needs; the `.hexpack` beside it holds the rest; the two belong together by a key stored in
both.

The engines are built on [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) with a Hexagon NPU backend
(`ggml-hexagon`), the same fork as [pulsex-image](https://github.com/Erik-matrix/pulsex-image) and
[pulsex-zimage](https://github.com/Erik-matrix/pulsex-zimage).

## Build

The binaries in `bin\` are prebuilt. `src\CMakeLists.txt` builds `wan-stream.exe` (against the ggml libraries of the
PulseX fork of ggml-hexagon) and the window `pulsex-wan.exe` (Dear ImGui + DX11): either inside the fork, as
`examples/wan22`, or on its own against an installed fork with `-Dggml_DIR=<fork install>/lib/cmake/ggml`, and
`-DWAN_IMGUI_DIR=<Dear ImGui>` for the window. The comments at the top of the file give the commands. The fork itself
is not in this repository.

## License

The code in this repository is released under the [MIT License](LICENSE); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for what it is built with and what it runs.

**The models are not MIT.** Wan 2.2 TI2V-5B (Wan-AI) and the files made from it on Hugging Face are under the
Apache License 2.0; the umT5 encoder (city96, from Google's umT5) is Apache-2.0; TAEHV (madebyollin) is MIT. Qualcomm's QNN runtime is under Qualcomm's own license and is never in
this repository or on Hugging Face. The videos you make are yours to use; you are responsible for them, and for
saying that they are AI-generated where the law requires it. This project is not affiliated with or endorsed by
Wan-AI, Alibaba or Qualcomm.
