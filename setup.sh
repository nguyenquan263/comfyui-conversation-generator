#!/usr/bin/env bash
# One-shot setup for a fresh clone of this repo (Linux / WSL2, NVIDIA GPU, no sudo needed).
#
#   ./setup.sh              # everything, including ~50 GB of model weights for all pipelines
#   ./setup.sh wan ltx25    # only the weights those pipelines need
#   ./setup.sh --no-models  # code + ComfyUI + venv only; run ./setup_models.py later
#
# What it does (each step is skipped when already done, so it is safe to re-run):
#   1. uv + a managed Python 3.12            (~/.local/bin/uv)
#   2. ComfyUI v0.41.0 + ComfyUI-GGUF node   (./ComfyUI, from GitHub release tarballs; no git needed)
#   3. venv on the Linux disk + requirements (~/.venvs/comfyui, linked here as ./venv)
#   4. model weights, SHA-256 verified       (./setup_models.py)
#   5. the editor workflows into ComfyUI's sidebar folder
#   6. a GPU smoke test with the same environment the app uses
set -euo pipefail

COMFY_VERSION="${COMFY_VERSION:-v0.41.0}"
VENV="${COMFY_VENV:-$HOME/.venvs/comfyui}"          # cartoon.py expects ~/.venvs/comfyui
cd "$(dirname "$0")"
ROOT=$(pwd)
export PATH="$HOME/.local/bin:$PATH"

PIPELINES=()
MODELS=1
for a in "$@"; do
    case "$a" in
        --no-models) MODELS=0 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) PIPELINES+=("$a") ;;
    esac
done

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
fetch_tarball() {  # url, destination dir: downloads and unpacks a GitHub tarball, stripping the top folder
    local url=$1 dest=$2 tmp
    tmp=$(mktemp -d)
    curl -L --fail --retry 5 --progress-bar -o "$tmp/src.tar.gz" "$url"
    mkdir -p "$dest"
    tar -xzf "$tmp/src.tar.gz" -C "$dest" --strip-components=1
    rm -rf "$tmp"
}

step "1/6 uv and Python 3.12"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv python install 3.12 >/dev/null
echo "uv $(uv --version | cut -d' ' -f2), python $(uv run --python 3.12 python --version 2>/dev/null | cut -d' ' -f2)"

step "2/6 ComfyUI $COMFY_VERSION and the ComfyUI-GGUF custom node"
if [ ! -f ComfyUI/main.py ]; then
    fetch_tarball "https://github.com/comfyanonymous/ComfyUI/archive/refs/tags/$COMFY_VERSION.tar.gz" ComfyUI
fi
if [ ! -f ComfyUI/custom_nodes/ComfyUI-GGUF/nodes.py ]; then
    fetch_tarball "https://github.com/city96/ComfyUI-GGUF/archive/refs/heads/main.tar.gz" ComfyUI/custom_nodes/ComfyUI-GGUF
fi
echo "ComfyUI $(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' ComfyUI/comfyui_version.py), GGUF node present"

step "3/6 virtualenv at $VENV"
# The venv lives on the Linux filesystem on purpose: a venv on a Windows-mounted drive (/mnt/*)
# takes an hour to install and makes every start slow.
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -r requirements.txt
ln -sfn "$VENV" venv
echo "torch $("$VENV/bin/python" -c 'import torch; print(torch.__version__)')"

step "4/6 model weights"
if [ "$MODELS" = 1 ]; then
    if [ ${#PIPELINES[@]} -eq 0 ]; then
        "$VENV/bin/python" setup_models.py all
    else
        "$VENV/bin/python" setup_models.py "${PIPELINES[@]}"
    fi
else
    echo "skipped (--no-models); run ./setup_models.py <pipeline>|all later"
fi

step "5/6 workflows for the ComfyUI sidebar"
mkdir -p ComfyUI/user/default/workflows
cp workflows/*.json ComfyUI/user/default/workflows/ 2>/dev/null && rm -f ComfyUI/user/default/workflows/*.api.json
ls ComfyUI/user/default/workflows

step "6/6 GPU smoke test"
# Same two workarounds cartoon.py applies at launch (see its server_env for why).
mkdir -p "$HOME/.comfy-nvml-shadow" && : > "$HOME/.comfy-nvml-shadow/libnvidia-ml.so.1"
if LD_LIBRARY_PATH="$HOME/.comfy-nvml-shadow${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" TORCH_DISABLE_NATIVE_JIT=1 \
   "$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.mem_get_info()[1]/2**30:.1f} GB")'; then
    :
else
    echo "WARNING: torch cannot see a CUDA GPU; the API will fail on the first job" >&2
fi
if grep -qi microsoft /proc/version 2>/dev/null; then
    mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    if [ "$mem_gb" -lt 9 ]; then   # memory=10GB in .wslconfig shows up as ~9.7 GB
        echo "NOTE: WSL has ${mem_gb} GB RAM. The LTX 2.5 pipelines need ~10 GB plus swap; put this in"
        echo "      %USERPROFILE%\\.wslconfig and run 'wsl --shutdown':   [wsl2]  memory=10GB  swap=16GB"
    fi
fi

printf '\n\033[1;32mdone.\033[0m  start the API with:  ./api.py      (web form: http://127.0.0.1:8000, ComfyUI: http://127.0.0.1:8188)\n'
