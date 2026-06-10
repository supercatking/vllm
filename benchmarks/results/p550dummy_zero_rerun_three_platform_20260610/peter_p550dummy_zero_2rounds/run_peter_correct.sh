#!/usr/bin/env bash
set -u
ROOT="$1"
SRC=/home/ubuntu/vllm_p550dummy_src
VENV=/home/ubuntu/vllm/.venv-p550
MODEL=/home/ubuntu/vllm/.p550_models/qwen2.5-0.5b-instruct
run_round() {
  local round="$1"
  local out="$ROOT/peter_round${round}"
  mkdir -p "$out"
  pkill -f 'VLLM::EngineCore|vllm.entrypoints.cli.main bench latency|vllm.entrypoints.cli.main bench synthetic' 2>/dev/null || true
  rm -f /dev/shm/psm_* /dev/shm/*vllm* 2>/dev/null || true
  sleep 2
  cd "$SRC"
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$SRC" \
  VLLM_USE_V1=1 \
  VLLM_TARGET_DEVICE=cpu \
  VLLM_WORKER_MULTIPROC_METHOD=fork \
  "$VENV/bin/python" -m vllm.entrypoints.cli.main bench synthetic \
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
    --gpu-memory-utilization 0.05 \
    --max-model-len 2048 \
    --output-dir "$out" \
    > "$out/bench.log" 2> "$out/bench.err"
  local rc=$?
  pkill -f 'VLLM::EngineCore|vllm.entrypoints.cli.main bench latency|vllm.entrypoints.cli.main bench synthetic' 2>/dev/null || true
  rm -f /dev/shm/psm_* /dev/shm/*vllm* 2>/dev/null || true
  return $rc
}
{
  echo "root=$ROOT"
  uname -a
  echo "python=$($VENV/bin/python --version 2>&1)"
  echo "source_dir=$SRC"
  echo "source_commit=$(cd $SRC && git rev-parse --short HEAD 2>/dev/null || echo archive)"
} > "$ROOT/env_report.txt"
for r in 1 2; do
  echo "round $r start $(date -Is)" | tee -a "$ROOT/run.log"
  run_round "$r"
  rc=$?
  echo "round $r rc=$rc end $(date -Is)" | tee -a "$ROOT/run.log"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
  "$VENV/bin/python" - <<PY >> "$ROOT/run.log"
import json, pathlib, statistics, math
p=pathlib.Path('$ROOT')/'peter_round${r}'/'matrix_results.json'
d=json.loads(p.read_text())
rows=d.get('cases') or []
ok=[x for x in rows if x.get('success')]
print('round${r}_rows', len(rows), 'ok', len(ok))
if ok:
    print('round${r}_avg_ttft', statistics.mean(x['ttft_avg_ms'] for x in ok))
    print('round${r}_avg_tpot', statistics.mean(x['tpot_ms'] for x in ok))
    print('round${r}_geomean_tps', math.exp(sum(math.log(x['decode_tps']) for x in ok)/len(ok)))
for x in rows:
    if not x.get('success'):
        print('round${r}_fail', x.get('case_id'), x.get('invalid_reason'))
PY
  sleep 3
done
