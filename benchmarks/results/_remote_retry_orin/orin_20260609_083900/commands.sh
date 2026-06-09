# Remote synthetic host benchmark retry on orin, all remote writes under:
#   /home/nvidia/vllm_synthetic_host_benchmark/orin_20260609_083900

ssh orin 'hostname; whoami; pwd; python3 --version; which python3'
ssh orin 'set -eu; echo HOST=$(hostname); echo USER=$(whoami); echo HOME=$HOME; echo ARCH=$(uname -m); echo KERNEL=$(uname -r); python3 --version; python3 -m pip --version || true; python3 -m venv --help >/dev/null 2>&1; echo VENV_STATUS=$?; df -h $HOME; command -v nvcc || true; command -v nvidia-smi || true; ls -la ~/vllm_synthetic_host_benchmark 2>/dev/null || true'
# Python probe confirmed: torch 2.5.0a0+872d972e41.nv24.08, CUDA 12.6, infer_schema False, transformers/vllm missing.

RUN_ID=orin_20260609_083900
REMOTE_BASE=/home/nvidia/vllm_synthetic_host_benchmark/$RUN_ID
ssh orin mkdir -p ${REMOTE_BASE}/src ${REMOTE_BASE}/results ${REMOTE_BASE}/logs ${REMOTE_BASE}/wheels
rsync -a --delete --exclude .deps --exclude __pycache__ --exclude '*.pyc' --exclude build --exclude dist --exclude .mypy_cache --exclude .pytest_cache --exclude .ruff_cache --exclude 'benchmarks/results/*' /home/zyz/vLLM_deploy/vllm/ orin:${REMOTE_BASE}/src/

# python3 -m venv failed because ensurepip/python3.10-venv is not installed and sudo was not allowed.
ssh orin mkdir -p ${REMOTE_BASE}/pytools
ssh orin python3 -m pip install --target ${REMOTE_BASE}/pytools virtualenv
ssh orin env PYTHONPATH=${REMOTE_BASE}/pytools python3 -m virtualenv --system-site-packages ${REMOTE_BASE}/venv

# Created sandbox-only vllm dist-info + compatibility shims under ${REMOTE_BASE}/venv and ${REMOTE_BASE}/shims.
ssh orin find ${REMOTE_BASE}/src/vllm -name '*.so' -type f -print -delete
ssh orin ${REMOTE_BASE}/venv/bin/python -m pip install 'transformers>=4.56.0,!=5.0.*,!=5.1.*,!=5.2.*,!=5.3.*,!=5.4.*,!=5.5.0' 'tokenizers>=0.21.1' 'safetensors>=0.6.2' 'pydantic>=2.12.0' 'protobuf>=5.29.6' blake3 py-cpuinfo msgspec pyyaml pyzmq filelock regex cachetools sentencepiece
ssh orin ${REMOTE_BASE}/venv/bin/python -m pip install -r ${REMOTE_BASE}/src/requirements/common.txt
ssh orin ${REMOTE_BASE}/venv/bin/python -m pip install numpy==1.26.4

# Final smoke attempt; failed before benchmark execution.
ssh orin env PYTHONPATH=${REMOTE_BASE}/shims:${REMOTE_BASE}/src VLLM_LOGGING_LEVEL=DEBUG ${REMOTE_BASE}/venv/bin/python -m vllm.entrypoints.cli.main bench synthetic --model Qwen/Qwen2.5-0.5B --input-lens 32 --output-lens 4 --warmup 0 --repeats 1 --max-model-len 128 --output-dir ${REMOTE_BASE}/results/smoke

# Matrix was skipped because smoke did not pass.
# Intended matrix command would have been:
ssh orin env PYTHONPATH=${REMOTE_BASE}/shims:${REMOTE_BASE}/src VLLM_LOGGING_LEVEL=INFO ${REMOTE_BASE}/venv/bin/python -m vllm.entrypoints.cli.main bench synthetic --model Qwen/Qwen2.5-0.5B --input-lens 32,128,512,1024 --output-lens 64,128,512 --warmup 1 --repeats 5 --max-model-len 2048 --output-dir ${REMOTE_BASE}/results/matrix