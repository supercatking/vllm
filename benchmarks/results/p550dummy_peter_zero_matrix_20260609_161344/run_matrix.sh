#!/usr/bin/env bash
set -euo pipefail
PY=/home/ubuntu/vllm/.venv-p550/bin/python
SRC=/home/ubuntu/vllm_p550dummy_src
OUT="$1"
cd /tmp
PYTHONPATH="$SRC" VLLM_USE_V1=1 VLLM_TARGET_DEVICE=cpu VLLM_WORKER_MULTIPROC_METHOD=fork \
  "$PY" -m vllm.entrypoints.cli.main bench synthetic \
  --model /home/ubuntu/vllm/.p550_models/qwen2.5-0.5b-instruct \
  --gpu-memory-utilization 0.05 \
  --dummy-prefill-ms 0 \
  --dummy-decode-ms 0 \
  --dummy-output-token-id 0 \
  --input-lens 32,128,512,1024 \
  --output-lens 64,128,512 \
  --warmup 1 \
  --repeats 5 \
  --max-model-len 2048 \
  --output-dir "$OUT" > "$OUT/run.log" 2>&1
