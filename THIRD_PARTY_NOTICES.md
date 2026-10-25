# Third-party notices

The PulseX code is MIT (see LICENSE). It is built with, or runs, the following, each under its own license.

## Included in this repository

- **llama.cpp / ggml** - MIT License, the ggml authors. `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`,
  `ggml-hexagon.dll`, `llama.dll`, `llama-common.dll`, `llama-embedding.exe` and the NPU modules
  `libggml-htp-v7x.so` (with their catalog `libggml-htp.cat`) are built from the PulseX fork of llama.cpp with a
  Hexagon NPU backend; `wan-stream.exe` is built against it. The license text is in `bin/LICENSE-llama.cpp-ggml.txt`.
- **Dear ImGui** - MIT License, Omar Cornut and contributors. Compiled into `pulsex-wan.exe`. The license text is in
  `bin/LICENSE-imgui.txt`.
- **TAEHV** (Tiny AutoEncoder for Hunyuan Video / Wan) - MIT License, Copyright (c) 2025 Ollin Boer Bohan.
  `third_party/taehv/taehv.py` is an unmodified copy of `taehv.py` from
  [madebyollin/taehv](https://github.com/madebyollin/taehv) (commit `011dfc2`, 2026-08-31); its license text is in
  `third_party/taehv/LICENSE`. It is the "Fast" decoder of the window.

## Not included (downloaded separately)

- **Wan 2.2 TI2V-5B** (Wan-AI, Alibaba) - Apache License 2.0. The video model this program runs, and its VAE. The
  converted files on Hugging Face (the slim GGUF + `.hexpack` of the transformer, the extras file, the VAE in Q4_0
  and the compiled NPU graphs of the VAE) are derivatives of it, modified by PulseX (quantised, repacked for the NPU,
  compiled), and are shared there under Apache-2.0 with its LICENSE and a NOTICE.
- **Wan 2.2 TI2V-5B, Q4_0 GGUF** (QuantStack/Wan2.2-TI2V-5B-GGUF) - Apache License 2.0. The quantised transformer the
  files above were made from.
- **umT5-XXL encoder, Q8_0 GGUF** (city96/umt5-xxl-encoder-gguf) - Apache License 2.0; from Google's umT5-XXL. The text
  encoder; downloaded unchanged from its own repository.
- **spiece.model** - the umT5 SentencePiece tokenizer (Google), Apache License 2.0, as shipped with Wan 2.2.
- **taew2_2_super.pth** - the weights of TAEHV for Wan 2.2, MIT License (madebyollin/taehv); downloaded from its
  own repository.
- **Qualcomm AI runtime (QAIRT / QNN)** - `QnnHtp.dll`, `QnnSystem.dll`, `QnnHtpV73Stub.dll`, `libQnnHtpV73Skel.so`
  and its catalog. Loaded at run time by ONNX Runtime's QNN execution provider for the VAE graphs. Under Qualcomm's own
  license (the Qualcomm AI Stack License); they are never in this repository. The setup file includes them, unchanged
  and in object code form, as part of the application (`qnn\ABOUT-QNN.txt` and Qualcomm's notice beside them).
- **ONNX Runtime with the QNN execution provider** (`onnxruntime-qnn` on PyPI) - MIT License (Microsoft); the wheel
  holds Qualcomm files under Qualcomm's terms. Installed by pip from PyPI, not shipped here.
- **PyTorch, diffusers, NumPy, gguf, SentencePiece, safetensors, Pillow, tqdm** - installed by pip
  (`requirements.txt`), each under its own license (BSD-style, Apache-2.0, MIT, HPND).
- **FFmpeg** - writes the mp4. Not in this repository and not inside the setup file: the setup downloads the
  Windows ARM64 build `ffmpeg-N-125875-g5d4d3bdc61-winarm64-gpl.zip` of BtbN/FFmpeg-Builds (release `autobuild-2026-07-31-14-10`,
  https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-07-31-14-10/ffmpeg-N-125875-g5d4d3bdc61-winarm64-gpl.zip), verifies it by SHA-256
  `b254b9248d9247d3c87f59ae6ec29fe12fe3644569ed9e04ca8265625e6f2fc9`, and installs only its `ffmpeg.exe`, beside that build's own
  license text (`bin/LICENSE-ffmpeg.txt`). That build is licensed under the GNU General Public License version 3
  (it is built with GPL components such as libx264); its sources are FFmpeg's (https://ffmpeg.org, commit
  `5d4d3bdc61`) and BtbN's build scripts (https://github.com/BtbN/FFmpeg-Builds). PulseX only runs it as a separate
  program. If you install another build yourself, its license is the one of that build.

No warranty. Nothing you type or generate leaves the computer: the PulseX programs and the pipeline make no network
connection. (`llama-common.dll`, llama.cpp's shared library behind `llama-embedding.exe`, carries llama.cpp's own
optional model-download code and so links the Windows socket library; the pipeline never asks it to download.)
