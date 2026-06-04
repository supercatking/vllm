#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# P550 offline Qwen deployment script v2.
set -euo pipefail

repo_dir="${P550_REPO_DIR:-$(pwd)}"
bundle_tar="${P550_BUNDLE_TAR:-$repo_dir/p550_vllm_qwen_offline_bundle_v2.tar.gz}"
work_dir="${P550_BUNDLE_WORK_DIR:-$repo_dir/.p550_offline_bundle}"
model_name="${P550_MODEL_NAME:-qwen2.5-0.5b-instruct}"
service_port="${PORT:-8000}"
omp_threads="${P550_OMP_NUM_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

cd "$repo_dir"

if [ ! -f "$bundle_tar" ]; then
    echo "Missing offline bundle: $bundle_tar" >&2
    exit 1
fi

mkdir -p "$work_dir"
tar -xzf "$bundle_tar" -C "$work_dir" --strip-components=1

runtime_libs="$work_dir/runtime_libs"
export LD_LIBRARY_PATH="$runtime_libs:${LD_LIBRARY_PATH:-}"

repair_prebuilt_venv_paths() {
    if [ ! -x .venv-p550/bin/python ]; then
        return
    fi

    .venv-p550/bin/python - <<PY
from pathlib import Path
import re

new_path = str(Path(r"$repo_dir") / ".venv-p550")
old_path_pattern = re.compile(r"/home/[^\s'\":]+/vllm/\.venv-p550")

for path in Path(".venv-p550/bin").iterdir():
    if not path.is_file() or path.is_symlink():
        continue
    try:
        data = path.read_text()
    except UnicodeDecodeError:
        continue
    updated = old_path_pattern.sub(new_path, data)
    if updated != data:
        path.write_text(updated)
        print(f"Relocated venv script: {path}")
PY
}

repair_pip_if_needed() {
    if .venv-p550/bin/python -m pip --version >/dev/null 2>&1; then
        return
    fi

    pip_wheel="$(find "$work_dir/wheelhouse" -maxdepth 1 -name 'pip-*.whl' \
        | sort | tail -n 1)"
    if [ -z "$pip_wheel" ]; then
        echo "pip is broken and no pip wheel is available in the wheelhouse." >&2
        exit 1
    fi

    echo "Repairing pip from $pip_wheel"
    .venv-p550/bin/python - <<PY
from pathlib import Path
import shutil
import zipfile

site = Path(".venv-p550/lib/python3.12/site-packages")
wheel = Path(r"$pip_wheel")
for item in site.glob("pip*"):
    if item.is_dir():
        shutil.rmtree(item)
    elif item.is_file():
        item.unlink()
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(site)
PY
}

install_runtime_patch_deps() {
    if [ ! -d "$work_dir/wheelhouse" ]; then
        return
    fi

    missing="$(
        .venv-p550/bin/python - <<'PY'
modules = {
    "requests": "requests==2.31.0",
    "certifi": "certifi",
    "idna": "idna",
    "urllib3": "urllib3",
    "charset_normalizer": "charset-normalizer",
    "psutil": "psutil==5.9.8",
    "six": "six==1.17.0",
    "distro": "distro==1.9.0",
}
missing = []
for module_name, package_spec in modules.items():
    try:
        __import__(module_name)
    except Exception:
        missing.append(package_spec)
print(" ".join(missing))
PY
    )"

    if [ -n "$missing" ]; then
        repair_pip_if_needed
        echo "Installing missing runtime Python packages: $missing"
        .venv-p550/bin/python -m pip install --no-index \
            --find-links "$work_dir/wheelhouse" $missing
    fi
}

ensure_vllm_source_visible() {
    .venv-p550/bin/python - <<PY
from pathlib import Path
import site

site_dir = Path(site.getsitepackages()[0])
(site_dir / "p550_vllm_source.pth").write_text(str(Path(r"$repo_dir")) + "\n")
print("vLLM source path registered:", site_dir / "p550_vllm_source.pth")
PY
}

ensure_numpy_available() {
    if .venv-p550/bin/python - <<'PY' >/dev/null 2>&1
import numpy
PY
    then
        return
    fi

    if [ "${P550_INSTALL_NUMPY_WITH_APT:-1}" != "1" ]; then
        echo "numpy is missing and P550_INSTALL_NUMPY_WITH_APT is disabled." >&2
        echo "Run: sudo apt-get install -y python3-numpy" >&2
        exit 1
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "numpy is missing and apt-get is not available." >&2
        echo "Install python3-numpy before running this script again." >&2
        exit 1
    fi

    echo "numpy is missing; installing python3-numpy with apt."
    if [ "${P550_APT_UPDATE:-0}" = "1" ]; then
        sudo apt-get update
    fi
    sudo apt-get install -y python3-numpy
}

ensure_pil_available() {
    if .venv-p550/bin/python - <<'PY' >/dev/null 2>&1
import PIL
PY
    then
        return
    fi

    if [ "${P550_INSTALL_PIL_WITH_APT:-1}" != "1" ]; then
        echo "PIL is missing and P550_INSTALL_PIL_WITH_APT is disabled." >&2
        echo "Install python3-pil or add a compatible Pillow package first." >&2
        exit 1
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "PIL is missing and apt-get is not available." >&2
        echo "Install python3-pil or add a compatible Pillow package first." >&2
        exit 1
    fi

    echo "PIL is missing; installing python3-pil with apt."
    if [ "${P550_APT_UPDATE:-0}" = "1" ]; then
        sudo apt-get update
    fi
    sudo apt-get install -y python3-pil
}

verify_text_runtime() {
    .venv-p550/bin/python - <<'PY'
missing = []
for module_name in ("numpy", "PIL", "transformers", "torch", "vllm"):
    try:
        __import__(module_name)
    except Exception as exc:
        missing.append(f"{module_name}: {type(exc).__name__}: {exc}")
if missing:
    print("Missing runtime modules:")
    for item in missing:
        print("  " + item)
    print("Install the board OS package for Pillow, for example python3-pil, "
          "or include a compatible Pillow wheel in the offline bundle.")
    raise SystemExit(1)
from vllm import LLM, SamplingParams  # noqa: F401
print("Runtime import check passed.")
PY
}

restore_prebuilt_env() {
    if [ -x .venv-p550/bin/python ] && \
       .venv-p550/bin/python - <<'PY' >/dev/null 2>&1
import torch
import vllm
import vllm._C
PY
    then
        echo "Existing .venv-p550 can import vLLM; keeping it."
        return
    fi

    if [ -d .venv-p550 ]; then
        backup=".venv-p550.bak.$(date +%Y%m%d%H%M%S)"
        echo "Backing up existing .venv-p550 to $backup"
        mv .venv-p550 "$backup"
    fi
    echo "Restoring validated P550 virtual environment."
    tar -xzf "$work_dir/prebuilt/p550_venv_runtime.tar.gz" -C "$repo_dir"
}

restore_model() {
    mkdir -p .p550_models
    rm -rf ".p550_models/$model_name"
    cp -a "$work_dir/models/$model_name" ".p550_models/$model_name"
    test -f ".p550_models/$model_name/model.safetensors"
    test -f ".p550_models/$model_name/tokenizer.json"
}

restore_prebuilt_artifacts() {
    tar -xzf "$work_dir/prebuilt/p550_prebuilt_vllm_artifacts.tar.gz" -C "$repo_dir"
    test -f vllm/_C.abi3.so
    test -f vllm/spinloop.abi3.so
}

install_from_wheelhouse_if_requested() {
    if [ "${P550_REBUILD_FROM_WHEELHOUSE:-0}" != "1" ]; then
        return
    fi
    echo "Rebuilding Python environment from offline wheelhouse."
    [ -d .venv-p550 ] && mv .venv-p550 ".venv-p550.bak.$(date +%Y%m%d%H%M%S)"
    python3 -m venv --system-site-packages .venv-p550
    . .venv-p550/bin/activate
    python -m pip install --no-index --find-links "$work_dir/wheelhouse" --no-deps \
        torch==2.4.1 tokenizers==0.23.0rc0 llguidance pyzmq pydantic-core \
        blake3==1.0.6 safetensors==0.4.3 sentencepiece==0.2.0 msgspec \
        cloudpickle pydantic annotated-types typing-inspection transformers \
        huggingface_hub cbor2 einops gguf ijson py-cpuinfo pybase64 \
        python-json-logger setproctitle prometheus-client fsspec networkx \
        sympy aiohttp fastapi uvicorn watchfiles openai tqdm jinja2 regex \
        packaging setuptools setuptools-scm setuptools-rust maturin wheel ninja
    VLLM_TARGET_DEVICE=cpu VLLM_RVV_VLEN=0 MAX_JOBS=4 \
        python -m pip install -e . --no-build-isolation --no-deps
}

start_service() {
    if [ -f /tmp/p550_qwen_vllm.pid ]; then
        old_pid="$(cat /tmp/p550_qwen_vllm.pid 2>/dev/null || true)"
        [ -n "$old_pid" ] && kill "$old_pid" 2>/dev/null || true
    fi

    . .venv-p550/bin/activate
    setsid env \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        VLLM_P550_HOME="$repo_dir" \
        VLLM_P550_MODEL="$repo_dir/.p550_models/$model_name" \
        VLLM_P550_SERVED_MODEL_NAME="$model_name" \
        VLLM_P550_MAX_MODEL_LEN=128 \
        VLLM_P550_KV_CACHE_BYTES=536870912 \
        VLLM_TARGET_DEVICE=cpu \
        VLLM_RVV_VLEN=0 \
        OMP_NUM_THREADS="$omp_threads" \
        VLLM_CPU_OMP_THREADS_BIND=nobind \
        VLLM_WORKER_MULTIPROC_METHOD=fork \
        PORT="$service_port" \
        bash tools/p550_start_vllm_service.sh \
        </dev/null >/tmp/p550_qwen_vllm.log 2>&1 &
    echo $! >/tmp/p550_qwen_vllm.pid
}

wait_for_health() {
    for _ in $(seq 1 300); do
        if .venv-p550/bin/python - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${service_port}/health", timeout=2).read()
PY
        then
            .venv-p550/bin/python - <<PY
import urllib.request
print(urllib.request.urlopen("http://127.0.0.1:${service_port}/health", timeout=5).read().decode())
PY
            return
        fi
        sleep 1
    done
    echo "Service did not become healthy. Last log lines:" >&2
    tail -n 120 /tmp/p550_qwen_vllm.log >&2 || true
    exit 1
}

restore_prebuilt_env
repair_prebuilt_venv_paths
install_runtime_patch_deps
ensure_vllm_source_visible
install_from_wheelhouse_if_requested
repair_prebuilt_venv_paths
install_runtime_patch_deps
ensure_vllm_source_visible
ensure_numpy_available
ensure_pil_available
verify_text_runtime
restore_model
restore_prebuilt_artifacts
start_service
wait_for_health

P550_SERVICE_URL="http://127.0.0.1:${service_port}" \
P550_MODEL_NAME="$model_name" \
    .venv-p550/bin/python "$work_dir/scripts/p550_validate_10_chat.py"

grep -E "CPU_ATTN|Triton|fla_stub|unsupported" /tmp/p550_qwen_vllm.log | tail -n 80 || true
echo "P550 offline vLLM deployment v2 completed."
