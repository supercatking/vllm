# vLLM Synthetic Host Scheduler Benchmark

This report measures CPU-visible vLLM host/scheduler overhead with synthetic GPU execution. It does not measure model quality, CPU model execution, or real accelerator kernel performance.

## Configuration

- model: `/home/eswin/vllm/.p550_models/qwen2.5-0.5b-instruct`
- backend: `sleep`
- prefill delay: `5.0 ms`
- decode delay: `1.0 ms`
- dummy output token: `0`
- batch size: `1`
- repeats/warmup: `1/1`

## Environment

- hostname: `rockos-eswin`
- machine: `riscv64`
- kernel: `Linux rockos-eswin 6.6.92-eic7702-2025.08 #2025.09.10.11.31+ SMP Wed Sep 10 11:33:20 UTC 2025 riscv64 GNU/Linux`
- vLLM commit: `fatal: not a git repository (or any of the parent directories): .git`

## Results

| case_id | input_tokens_target | output_tokens_target | success | avg_e2e_ms | ttft_avg_ms | tpot_ms | decode_tps | host_adjusted_e2e_ms | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| i32_o3 | 32 | 3 | False |  |  |  |  |  | latency command exited 1 |

## Score

Reference-based scoring is optional.

```json
{
  "available": false,
  "reason": "no reference-json provided"
}
```

## Interpretation

- TTFT tracks the CPU-visible completion of synthetic prefill.
- TPOT and decode TPS track CPU-visible synthetic decode token return.
- host_adjusted_e2e_ms subtracts observed synthetic device delay from raw end-to-end latency to approximate host overhead.
