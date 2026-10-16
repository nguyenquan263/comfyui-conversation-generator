#!/home/qnguyen3/.venvs/comfyui/bin/python
"""
cartoon.py - one small app to run ComfyUI and render 5-second cartoons.

    cartoon.py render "a cat surfing a wave"      # render (starts the server if needed)
    cartoon.py serve                               # run ComfyUI in the foreground
    cartoon.py status                              # server / GPU / queue
    cartoon.py stop                                # stop a server this app started
    cartoon.py workflow > my.json                  # dump the API-format workflow

Only the standard library is used, so it runs under the ComfyUI venv python.
Models: Wan 2.1 T2V 1.3B + umt5-xxl fp8 + Wan 2.1 VAE, plus the CivitAI
"Flat Color - Style" LoRA (trigger words: "flat color, no lineart").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMFY_DIR = ROOT / "ComfyUI"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "comfyui.log"
PID_FILE = LOG_DIR / "comfyui.pid"
VENV_PY = Path.home() / ".venvs" / "comfyui" / "bin" / "python"
NVML_SHADOW = Path.home() / ".comfy-nvml-shadow"

HOST = "127.0.0.1"
PORT = 8188

# ComfyUI flags this machine needs:
#  --disable-pinned-memory / --disable-fast-disk: the pinned-memory file reader fails on the
#    Windows-mounted D: drive ("HostBuffer.read_file_slice failed").
#  --disable-comfy-compiler / --disable-triton-backend: there is no C compiler on this box (no sudo),
#    and Triton needs one to build kernels; without these the Flux 2 / LTX text encoders fail with
#    "Failed to find C compiler".
COMFY_ARGS = ["--disable-pinned-memory", "--disable-fast-disk",
              "--disable-comfy-compiler", "--disable-triton-backend"]

# English rendering of Wan's official default negative prompt (the model ships with a Chinese one;
# its umt5 encoder understands both).
DEFAULT_NEGATIVE = (
    "overly bright colors, overexposed, static, blurry details, subtitles, style, artwork, painting, "
    "still image, overall grayish, worst quality, low quality, JPEG artifacts, ugly, incomplete, "
    "extra fingers, badly drawn hands, badly drawn face, deformed, disfigured, malformed limbs, "
    "fused fingers, motionless frame, cluttered background, three legs, many people in background, "
    "walking backwards"
)
DEFAULT_PROMPT = (
    "A cheerful orange cat wearing a tiny blue scarf walks through a sunny meadow full of "
    "flowers, tail swishing, clouds drifting slowly in the pastel sky, smooth animation"
)
STYLE_PREFIX = "flat color, no lineart, cartoon style. "


# ----------------------------------------------------------------------------- server

def server_env() -> dict[str, str]:
    """Environment for ComfyUI with the WSL NVML workaround applied.

    On this machine libnvidia-ml.so.1 (NVML) makes ld.so abort with
    "dl-setup_hash.c: 36 ... Assertion failed" (nvidia-smi segfaults for the same reason).
    torch only uses NVML for optional stats, so an empty file shadowing it makes the dlopen
    fail cleanly and torch falls back to the CUDA driver API.
    """
    NVML_SHADOW.mkdir(parents=True, exist_ok=True)
    (NVML_SHADOW / "libnvidia-ml.so.1").touch()
    env = dict(os.environ)
    # torch 2.14 routes some ops (e.g. the rotary-embedding outer product used by the Flux 2 / LTX
    # text encoders) to Triton kernels, and Triton needs a C compiler this box does not have.
    # This switch keeps torch on its built-in kernels.
    env.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(NVML_SHADOW) + (":" + prev if prev else "")
    return env


def server_cmd(extra: list[str], port: int) -> list[str]:
    return [str(VENV_PY), "main.py", "--listen", HOST, "--port", str(port), *COMFY_ARGS, *extra]


def api(path: str, data: dict | None = None, port: int = PORT, timeout: float = 10):
    url = f"http://{HOST}:{port}{path}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def server_up(port: int = PORT) -> bool:
    try:
        api("/system_stats", port=port, timeout=3)
        return True
    except Exception:
        return False


def read_pid() -> int | None:
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def start_server_background(port: int, extra: list[str]) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    log = open(LOG_FILE, "ab")
    proc = subprocess.Popen(
        server_cmd(extra, port), cwd=COMFY_DIR, env=server_env(),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    print(f"starting ComfyUI (pid {proc.pid}, log {LOG_FILE}) ...", end="", flush=True)
    for _ in range(120):
        if server_up(port):
            print(" ready")
            return
        if proc.poll() is not None:
            print("\nComfyUI exited during startup. Last log lines:")
            print(tail_log(30))
            sys.exit(1)
        print(".", end="", flush=True)
        time.sleep(2)
    print("\nComfyUI did not become ready in time; see", LOG_FILE)
    sys.exit(1)


def ensure_server(port: int) -> None:
    if not server_up(port):
        start_server_background(port, [])


def tail_log(lines: int = 20) -> str:
    try:
        data = LOG_FILE.read_bytes()[-20000:].decode("utf-8", "replace")
    except FileNotFoundError:
        return ""
    return "\n".join(data.replace("\r", "\n").splitlines()[-lines:])


def last_progress() -> str:
    """Latest tqdm fragment from the server log, e.g. '40%|####  | 12/30 [01:52<02:48, 9.4s/it]'."""
    try:
        with open(LOG_FILE, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4000))
            chunk = f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""
    for frag in reversed(chunk.replace("\n", "\r").split("\r")):
        if "%|" in frag and "it" in frag:
            return frag.strip()
    return ""


# ----------------------------------------------------------------------------- workflow

def build_workflow(a: argparse.Namespace) -> dict:
    prompt = (STYLE_PREFIX if not a.no_style else "") + a.prompt
    seed = a.seed if a.seed is not None else random.randint(0, 2**32 - 1)
    return {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "wan2.1_t2v_1.3B_fp16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0], "lora_name": "wan_flat_color_1.3b_v2.safetensors",
            "strength_model": a.lora_strength}},
        "3": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": a.negative}},
        "7": {"class_type": "EmptyHunyuanLatentVideo", "inputs": {
            "width": a.width, "height": a.height, "length": a.frames, "batch_size": 1}},
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": a.shift}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["8", 0], "seed": seed, "steps": a.steps, "cfg": a.cfg,
            "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0,
            "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["4", 0]}},
        "11": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "fps": float(a.fps)}},
        # SaveVideo uses nested "dynamic combo" inputs; the API form is dotted keys.
        "12": {"class_type": "SaveVideo", "inputs": {
            "video": ["11", 0], "filename_prefix": f"video/{a.name}",
            "format": "mp4", "format.codec": "h264",
            "format.codec.encoding": "re-encode", "format.codec.encoding.crf": float(a.crf)}},
    }


# ----------------------------------------------------------------------------- commands

def cmd_serve(a: argparse.Namespace) -> None:
    if server_up(a.port):
        print(f"a server is already running on port {a.port}")
        return
    print(f"ComfyUI on http://{HOST}:{a.port}  (Ctrl+C to stop)")
    os.chdir(COMFY_DIR)
    os.execve(str(VENV_PY), server_cmd(a.extra, a.port), server_env())


def cmd_render(a: argparse.Namespace) -> None:
    if not a.no_autostart:
        ensure_server(a.port)
    elif not server_up(a.port):
        sys.exit(f"no server on port {a.port}; run `cartoon.py serve` or drop --no-autostart")

    wf = build_workflow(a)
    seed = wf["9"]["inputs"]["seed"]
    seconds = a.frames / a.fps
    print(f"prompt : {wf['5']['inputs']['text']}")
    print(f"video  : {a.width}x{a.height}, {a.frames} frames @ {a.fps} fps = {seconds:.1f}s, "
          f"{a.steps} steps, seed {seed}")

    try:
        pid = api("/prompt", {"prompt": wf}, port=a.port)["prompt_id"]
    except urllib.error.HTTPError as e:
        sys.exit(f"ComfyUI rejected the workflow: {e.read().decode(errors='replace')[:1500]}")
    print(f"queued {pid}  (watch at http://{HOST}:{a.port})")

    t0 = time.time()
    while True:
        time.sleep(2)
        hist = api(f"/history/{pid}", port=a.port)
        if pid in hist:
            break
        prog = last_progress()
        print(f"\r  {time.time() - t0:5.0f}s  {prog:<60}", end="", flush=True)
    print()

    entry = hist[pid]
    if entry["status"].get("status_str") == "error":
        for m in entry["status"].get("messages", []):
            if m[0] == "execution_error":
                sys.exit(f"render failed in node {m[1].get('node_type')}: {m[1].get('exception_message')}")
        sys.exit("render failed; see " + str(LOG_FILE))

    for out in entry["outputs"].values():
        for item in out.get("images", []) + out.get("video", []) + out.get("videos", []):
            sub = item.get("subfolder", "")
            path = COMFY_DIR / "output" / sub / item["filename"]
            print(f"saved  : {path}   ({time.time() - t0:.0f}s)")


def cmd_status(a: argparse.Namespace) -> None:
    pid = read_pid()
    print(f"server : {'up' if server_up(a.port) else 'down'} on http://{HOST}:{a.port}"
          + (f"  (pid {pid}, started by this app)" if pid else ""))
    if not server_up(a.port):
        return
    stats = api("/system_stats", port=a.port)
    for d in stats.get("devices", []):
        free, total = d["vram_free"] / 2**30, d["vram_total"] / 2**30
        print(f"gpu    : {d['name']}  vram {free:.1f} / {total:.1f} GB free")
    q = api("/queue", port=a.port)
    print(f"queue  : {len(q['queue_running'])} running, {len(q['queue_pending'])} pending")
    prog = last_progress()
    if q["queue_running"] and prog:
        print(f"progress: {prog}")
    vids = sorted(glob.glob(str(COMFY_DIR / "output" / "video" / "*.mp4")), key=os.path.getmtime)
    if vids:
        print(f"latest : {vids[-1]}")


def cmd_stop(a: argparse.Namespace) -> None:
    pid = read_pid()
    if pid is None:
        print("no server started by this app is running"
              + (" (but something answers on the port)" if server_up(a.port) else ""))
        return
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    for _ in range(20):
        if read_pid() is None:
            break
        time.sleep(0.5)
    PID_FILE.unlink(missing_ok=True)
    print(f"stopped pid {pid}")


def cmd_workflow(a: argparse.Namespace) -> None:
    json.dump(build_workflow(a), sys.stdout, indent=1, ensure_ascii=False)
    print()


# ----------------------------------------------------------------------------- cli

def add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="what happens in the cartoon")
    p.add_argument("--negative", default=DEFAULT_NEGATIVE, help="negative prompt")
    p.add_argument("--no-style", action="store_true", help="don't prepend the flat-color trigger words")
    p.add_argument("--seed", type=int, help="fixed seed (default: random)")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=6.0)
    p.add_argument("--shift", type=float, default=8.0)
    p.add_argument("--lora-strength", type=float, default=1.0)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frames", type=int, default=81, help="81 frames @ 16 fps = 5 s (use 4k+1)")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--crf", type=int, default=18, help="h264 quality, lower = better")
    p.add_argument("--name", default="cartoon", help="output filename prefix")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run ComfyUI in the foreground")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra ComfyUI flags, e.g. --lowvram")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("render", help="render a cartoon (starts the server if needed)")
    add_render_args(p)
    p.add_argument("--no-autostart", action="store_true")
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("status", help="server, GPU and queue status")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("stop", help="stop the server this app started")
    p.set_defaults(fn=cmd_stop)

    p = sub.add_parser("workflow", help="print the API-format workflow JSON")
    add_render_args(p)
    p.set_defaults(fn=cmd_workflow)

    a = ap.parse_args(argv)
    if a.cmd in ("render", "workflow") and (a.frames - 1) % 4:
        ap.error("--frames must be 4k+1 (e.g. 33, 49, 81)")
    a.fn(a)


if __name__ == "__main__":
    main()
