#!/usr/bin/env bash
set -euo pipefail
PY=/home/eswin/vllm/.venv-p550/bin/python
OUT="$1"
cd /tmp
VLLM_WORKER_MULTIPROC_METHOD=fork VLLM_USE_V1=1 VLLM_TARGET_DEVICE=cpu \
  "$PY" -m vllm.entrypoints.cli.main bench synthetic \
  --model /home/eswin/vllm/.p550_models/qwen2.5-0.5b-instruct \
  --gpu-memory-utilization 0.05 \
  --dummy-prefill-ms 5 \
  --dummy-decode-ms 1 \
  --dummy-output-token-id 0 \
  --input-lens 32,128,512,1024 \
  --output-lens 64,128,512 \
  --warmup 0 \
  --repeats 1 \
  --output-dir "$OUT" > "$OUT/run.log" 2>&1
