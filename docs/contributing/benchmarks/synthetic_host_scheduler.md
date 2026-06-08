# Synthetic Host Scheduler Benchmark

`vllm bench synthetic` measures the CPU-visible vLLM host path while replacing real accelerator execution with deterministic synthetic delays. It is designed for host scheduler and control-plane evaluation. It is not a CPU inference benchmark, and it is not a real GPU or NPU kernel benchmark.

## What It Measures

The benchmark keeps the vLLM V1 request path active: prompt construction, engine enqueue, scheduling, model-runner invocation, sampled token return, and output validation. The model-runner path uses dummy GPU execution underneath `vllm bench latency --dummy-gpu-execution`.

The synthetic backend currently supports `sleep` mode:

- prefill sleeps for `--dummy-prefill-ms`, default `5.0` ms;
- decode sleeps for `--dummy-decode-ms`, default `1.0` ms;
- every generated token must equal `--dummy-output-token-id`;
- every completion must have exactly the requested output length.

This separation is important on systems that also have a CPU backend deployment. A CPU backend run measures CPU model execution. This synthetic benchmark measures host scheduling overhead with accelerator compute replaced by dummy delay.

## Example

```bash
vllm bench synthetic   --model Qwen/Qwen2.5-0.5B-Instruct   --input-lens 32,128,512,1024   --output-lens 64,128,512   --batch-size 1   --concurrency 1   --warmup 1   --repeats 5   --dummy-prefill-ms 5   --dummy-decode-ms 1   --dummy-output-token-id 0   --gpu-memory-utilization 0.5   --output-dir /tmp/vllm_synthetic_host_scheduler
```

The command writes:

- `env_report.json`
- `run_config.json`
- `matrix_results.json`
- `matrix_results.csv`
- `analysis.md`
- `raw_logs/` with one raw latency JSON and log per matrix cell

## Metrics

- `avg_e2e_ms`: raw end-to-end latency from the wrapped latency benchmark.
- `ttft_avg_ms`: CPU-visible prefill completion latency.
- `tpot_ms`: decode latency per generated decode token.
- `decode_tps`: decode tokens divided by total decode latency.
- `prefill_*` and `decode_*`: avg/p50/p90/p99 synthetic step latency.
- `host_adjusted_e2e_ms`: raw E2E latency minus observed synthetic device delay.
- `host_adjusted_ttft_ms`: TTFT minus observed synthetic prefill delay.

The host-adjusted values are approximations. They are useful for comparing scheduler/control-plane cost across platforms, not for estimating real model latency.

## Validation Rules

A case is valid only when the wrapped latency benchmark succeeds and validates:

- all generated token IDs equal `--dummy-output-token-id`;
- each completion has exactly the requested output length;
- prefill latency exists;
- decode latency, TPOT, and decode TPS exist for output lengths greater than one.

Failed cases remain in the JSON/CSV/Markdown output with `success=false` and `invalid_reason` populated.

## Current Limitations

- v0.1 supports `--concurrency 1` only.
- v0.1 supports the `sleep` synthetic backend only.
- The implementation targets the V1 GPU runner dummy execution path.
- Serving/OpenAI HTTP request handling is not included in v0.1.
