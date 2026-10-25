# Every folder PulseX Wan's Python reads or writes,
# in ONE place. Per key: the environment variable, else wan.json beside this file ({"paths": {...}}; a relative path
# counts from wan.json's folder), else the installed layout (DEFAULT below).
#   MODELS      PULSEX_WAN_MODELS      "models"  the run files: DiT slim GGUF + hexpack, extras, umT5, spiece, VAE + its caches,
#                                                the QNN contexts, taehv\, vae_config\
#   BIN         PULSEX_WAN_BIN         "bin"     wan-stream.exe, llama-embedding.exe (+ the ggml DLLs)
#   QNN         PULSEX_QNN             "qnn"     the QNN runtime folder; QNN_HTP = its QnnHtp.dll (onnxruntime-qnn's backend_path)
#   FFMPEG      PULSEX_FFMPEG          "ffmpeg"  ffmpeg.exe (when that file is missing: ffmpeg on PATH, if there is one)
#   WORK        PULSEX_WAN_DATA        "work"    scratch (wan-stream's exchange files, the VAE's spill, the frame hand-over,
#                                                umT5's temp folder): <PULSEX_WAN_DATA>\work, as the window's own work folder
#   OUT         PULSEX_WAN_OUT         "out"     wan_pipeline.py's default --outdir. NEVER created here (only when written to)
#   SRC         PULSEX_WAN_SRC         "src"     a source folder (full GGUFs, safetensors): tools and fallbacks only
#   TEXT_CACHE  PULSEX_WAN_TEXT_CACHE  "text_cache"  umT5's output per prompt: <WORK>\text_cache
#                                                (the prompts stay out of the model folder)
# Nothing here creates a folder or touches a file at import (wan.json is only read).
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSON = Path(os.environ.get("PULSEX_WAN_JSON") or (HERE / "wan.json"))

# the installed layout: this file sits in <app>\python, beside <app>\bin, <app>\models and <app>\qnn
APP = HERE.parent
DEFAULT = {
    "models": str(APP / "models"),
    "bin": str(APP / "bin"),
    "qnn": str(APP / "qnn"),
    "ffmpeg": str(APP / "bin" / "ffmpeg.exe"),                          # missing: ffmpeg on PATH (below)
    "work": str(APP / "data" / "work"),                                 # = the window's own work folder (<app>\data\work)
    "out": os.path.expandvars(r"%USERPROFILE%\Videos\PulseX Wan"),      # = the window's films
    "src": str(APP / "models"),                                         # an installation has no separate source folder
}


def _read_json():
    try:
        js = json.loads(JSON.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:                                  # a wan.json that is there but unreadable: say so
        raise SystemExit("pulsex_paths: %s could not be read: %s" % (JSON, e))
    p = js.get("paths", js) if isinstance(js, dict) else None
    if not isinstance(p, dict):
        raise SystemExit("pulsex_paths: %s has no \"paths\" object" % JSON)
    return p


_J = _read_json()
SOURCE = {}                                                             # key -> where its value came from (env / wan.json / default)


def _get(key, env, default, env_sub=None):
    v = os.environ.get(env)
    if v:
        SOURCE[key] = "env " + env
        return Path(v) / env_sub if env_sub else Path(v)
    v = _J.get(key)
    if v:
        SOURCE[key] = "wan.json"
        p = Path(os.path.expandvars(v))
        return p if p.is_absolute() else (JSON.parent / p)
    SOURCE[key] = "default"
    return Path(default)


MODELS = _get("models", "PULSEX_WAN_MODELS", DEFAULT["models"])
BIN = _get("bin", "PULSEX_WAN_BIN", DEFAULT["bin"])
QNN = _get("qnn", "PULSEX_QNN", DEFAULT["qnn"])
QNN_HTP = QNN / "QnnHtp.dll"
FFMPEG = _get("ffmpeg", "PULSEX_FFMPEG", DEFAULT["ffmpeg"])
if not FFMPEG.exists() and shutil.which("ffmpeg"):                     # a machine without our copy: the one on PATH
    FFMPEG, SOURCE["ffmpeg"] = Path(shutil.which("ffmpeg")), SOURCE["ffmpeg"] + " -> PATH"
WORK = _get("work", "PULSEX_WAN_DATA", DEFAULT["work"], env_sub="work")
OUT = _get("out", "PULSEX_WAN_OUT", DEFAULT["out"])
SRC = _get("src", "PULSEX_WAN_SRC", DEFAULT["src"])
TEXT_CACHE = _get("text_cache", "PULSEX_WAN_TEXT_CACHE",
                  WORK / "text_cache")


def model_file(*parts):
    """a run file under MODELS, else the same name under SRC (PULSEX_WAN_SRC / wan.json "src")"""
    p = MODELS.joinpath(*parts)
    return p if p.exists() else SRC.joinpath(*parts)


def qnn_options(**extra):
    """onnxruntime-qnn's provider_options: our QNN runtime (the pip wheel's own QNN is too old for the contexts)"""
    return [dict({"backend_path": str(QNN_HTP), "htp_performance_mode": "burst"}, **extra)]


QNN_FILES = ("QnnHtp.dll", "QnnSystem.dll", "QnnHtpV73Stub.dll", "libQnnHtpV73Skel.so", "libqnnhtpv73.cat")
_IDS = {}


def _build_id(p):
    """the QNN build id written in a QNN file (v2.45.0.260326154327 ...), None if unreadable / none"""
    p = str(p)
    if p not in _IDS:
        import re
        try:
            m = re.search(rb"v2\.\d+\.\d+\.\d{12,}", Path(p).read_bytes())
            _IDS[p] = m.group().decode() if m else None
        except OSError:
            _IDS[p] = None
    return _IDS[p]


def _loaded_modules():
    """the full path of every module loaded in THIS process (EnumProcessModulesEx + GetModuleFileNameW)"""
    import ctypes
    import ctypes.wintypes as wt
    k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.EnumProcessModulesEx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), wt.DWORD]
    k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, wt.DWORD]
    h = ctypes.c_void_p(k32.GetCurrentProcess())
    n = 1024
    while True:
        arr, need = (ctypes.c_void_p * n)(), wt.DWORD()
        if not psapi.EnumProcessModulesEx(h, arr, ctypes.sizeof(arr), ctypes.byref(need), 3):   # LIST_MODULES_ALL
            return []
        if need.value <= ctypes.sizeof(arr):
            break
        n = need.value // ctypes.sizeof(ctypes.c_void_p) + 64
    buf, out = ctypes.create_unicode_buffer(32768), []
    for i in range(need.value // ctypes.sizeof(ctypes.c_void_p)):
        if arr[i] and k32.GetModuleFileNameW(arr[i], buf, 32768):
            out.append(buf.value)
    return out


def qnn_witness(what):
    """2026-10-25: call right after a QNN session is made. Says which QNN files THIS process really uses - from its
    loaded modules, not from what exists - in one line: "QNN (what): <dir> (stub ..., skel ...)". QnnHtp.dll falls
    back, with no error, to the NPU driver's own HTP files (DriverStore ...\\HTP: HtpUsrDrv.dll, QnnHtpV73StubDrv.dll,
    libQnnHtpV73SkelDrv.so) when our stub is missing or our skel is missing or refused - QnnHtp.dll accepts only a skel
    with its own build id. Then, or when one of our five files is missing, or the stub came from outside our QNN
    folder: a WARNING line naming what was used instead. Returns True when our own files are the ones in use."""
    import os
    mods = _loaded_modules()
    by = {}
    for m in mods:
        by.setdefault(os.path.basename(m).lower(), []).append(m)
    qdir = os.path.normcase(os.path.abspath(str(QNN)))
    ours = lambda p: os.path.normcase(os.path.dirname(os.path.abspath(p))) == qdir
    htp = by.get("qnnhtp.dll", [])
    stub = by.get("qnnhtpv73stub.dll", [])
    drv = by.get("htpusrdrv.dll", []) + by.get("qnnhtpv73stubdrv.dll", [])
    # where our skel / catalogue are: the QNN folder first, else a folder on ADSP_LIBRARY_PATH (the developer setup)
    dirs = [str(QNN)] + [d for d in os.environ.get("ADSP_LIBRARY_PATH", "").split(";") if d.strip()]
    found = {f: next((os.path.join(d, f) for d in (dirs if f.endswith((".so", ".cat")) else dirs[:1])
                      if os.path.isfile(os.path.join(d, f))), None) for f in QNN_FILES}
    skel = found["libQnnHtpV73Skel.so"]
    sid, hid = (_build_id(skel) if skel else None), (_build_id(htp[0]) if htp else None)
    why = []
    missing = [f for f, p in found.items() if p is None]
    if missing:
        why.append("missing: %s" % ", ".join(missing))
    if not htp:
        why.append("QnnHtp.dll is not loaded")
    elif not ours(htp[0]):
        why.append("QnnHtp.dll from %s" % htp[0])
    if not stub:
        why.append("our QnnHtpV73Stub.dll is not loaded")
    elif not ours(stub[0]):
        why.append("the stub came from %s" % stub[0])
    if drv:
        why.append("the NPU driver's own HTP files are in use: %s (build %s)" % (
            ", ".join(drv), _build_id(os.path.join(os.path.dirname(drv[0]), "libQnnHtpV73SkelDrv.so")) or "?"))
        if skel and sid and hid and sid != hid:
            why.append("our skel %s is build %s, QnnHtp.dll %s accepts only its own build id" % (skel, sid, hid))
    line = "QNN (%s): %s (stub %s, skel %s%s)" % (what, QNN, stub[0] if stub else "-", skel or "-", " " + sid if sid else "")
    print(line, flush=True)
    for w in why:
        print("WARNING: QNN (%s): %s" % (what, w), flush=True)
    return not why


def work_dir(*parts):
    """a scratch folder under WORK, created when asked for"""
    d = WORK.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    for k in ("MODELS", "BIN", "QNN", "QNN_HTP", "FFMPEG", "WORK", "OUT", "SRC", "TEXT_CACHE"):
        key = k.lower().replace("qnn_htp", "qnn")
        print("%-10s %-55s (%s)" % (k, globals()[k], SOURCE.get(key, "")))
