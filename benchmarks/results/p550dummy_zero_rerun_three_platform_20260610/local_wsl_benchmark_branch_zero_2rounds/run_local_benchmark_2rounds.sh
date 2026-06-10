#!/usr/bin/env bash
set -euo pipefail
REPO=/home/zyz/vLLM_deploy/vllm
PY=/home/zyz/vLLM_deploy/.venv/bin/python
MODEL=/home/zyz/vLLM_deploy/localModels/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
ROOT="$1"
cd "$REPO"
for round in 1 2; do
  OUT="$ROOT/local_round${round}"
  mkdir -p "$OUT"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=fork \
    "$PY" -m vllm.entrypoints.cli.main bench synthetic \
    --model "$MODEL" \
    --input-lens 32,128,512,1024 \
    --output-lens 64,128,512 \
    --batch-size 1 \
    --concurrency 1 \
    --warmup 1 \
    --repeats 5 \
    --dummy-prefill-ms 0 \
    --dummy-decode-ms 0 \
    --dummy-output-token-id 0 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 2048 \
    --output-dir "$OUT" > "$OUT/run.log" 2>&1
done
