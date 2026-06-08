# V1 Dummy GPU Latency Benchmark

This benchmark mode measures the CPU-visible latency of the V1 GPU execution
path without running real model kernels. It is intended for scheduler and CPU
load evaluation, not model quality or GPU kernel benchmarking.

## Behavior

Enable the mode with `vllm bench latency --dummy-gpu-execution`.

When enabled:

- V1 GPU prefill execution sleeps for `--dummy-prefill-delay-ms` milliseconds.
- V1 GPU decode execution sleeps for `--dummy-decode-delay-ms` milliseconds.
- Every generated token is forced to `--dummy-token-id`, default `0`.
- `vllm bench latency` validates that every output token is the dummy token and
  that every completion has exactly `--output-len` tokens.
- Async scheduling is disabled for this benchmark mode so the CPU-visible token
  completion point is unambiguous.

The mode is benchmark-only. Normal `vllm bench latency` and serving paths are
unchanged unless `--dummy-gpu-execution` is explicitly provided.

## Single Case

```bash
vllm bench latency \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --load-format dummy \
  --enforce-eager \
  --input-len 1024 \
  --output-len 512 \
  --batch-size 1 \
  --gpu-memory-utilization 0.5 \
  --num-iters-warmup 1 \
  --num-iters 5 \
  --dummy-gpu-execution \
  --dummy-prefill-delay-ms 5 \
  --dummy-decode-delay-ms 1 \
  --dummy-token-id 0 \
  --output-json /tmp/dummy_gpu_latency_i1024_o512.json
```

The JSON output embeds a `dummy_gpu_latency` object with:

- `ttft_ms`: prefill latency statistics.
- `tpot_ms`: total decode latency divided by generated decode tokens.
- `decoding_tps`: generated decode tokens divided by total decode latency.
- `prefill_latency_ms`: prefill and chunked-prefill step statistics.
- `decode_latency_ms`: decode step latency statistics.
- `total_step_latency_ms`: all dummy GPU steps.
- `step_records`: raw per-step records.

## Latency Matrix

Use the helper below to generate a table across input and output lengths:

```bash
python benchmarks/run_dummy_gpu_latency_matrix.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input-lens 32,128,512,1024 \
  --output-lens 64,128,512 \
  --batch-size 1 \
  --gpu-memory-utilization 0.5 \
  --num-iters-warmup 1 \
  --num-iters 5 \
  --dummy-prefill-delay-ms 5 \
  --dummy-decode-delay-ms 1 \
  --output-dir /tmp/vllm_dummy_gpu_latency_matrix
```

The helper writes:

- `dummy_gpu_latency_matrix.json`
- `dummy_gpu_latency_matrix.csv`
- `dummy_gpu_latency_matrix.md`
- one raw `vllm bench latency` JSON file per matrix cell

Use the matrix to estimate CPU scheduler overhead under controlled dummy GPU
latency. TTFT reflects CPU-visible prefill completion. TPOT and decode TPS
reflect CPU-visible decode token completion after the first token.

## Pass Criteria

A dummy GPU benchmark run is valid only when:

- every generated output token equals `--dummy-token-id`;
- every completion has exactly `--output-len` tokens;
- TTFT is present when prefill occurs;
- TPOT and decode TPS are present when `--output-len > 1`;
- `prefill_latency_ms` and `decode_latency_ms` are present in the output JSON
  for workloads that exercise both phases.
