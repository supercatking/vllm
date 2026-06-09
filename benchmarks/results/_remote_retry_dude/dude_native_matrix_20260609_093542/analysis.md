# vLLM Synthetic Host Scheduler Benchmark

This report measures CPU-visible vLLM host/scheduler overhead with synthetic GPU execution. It does not measure model quality, CPU model execution, or real accelerator kernel performance.

## Configuration

- model: `/home/eswin/vllm/.p550_models/qwen2.5-0.5b-instruct`
- backend: `sleep`
- prefill delay: `5.0 ms`
- decode delay: `1.0 ms`
- dummy output token: `0`
- batch size: `1`
- repeats/warmup: `1/0`

## Environment

- hostname: `rockos-eswin`
- machine: `riscv64`
- kernel: `Linux rockos-eswin 6.6.92-eic7702-2025.08 #2025.09.10.11.31+ SMP Wed Sep 10 11:33:20 UTC 2025 riscv64 GNU/Linux`
- vLLM commit: `fatal: not a git repository (or any parent up to mount point /)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).`

## Results

| case_id | input_tokens_target | output_tokens_target | success | avg_e2e_ms | ttft_avg_ms | tpot_ms | decode_tps | host_adjusted_e2e_ms | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| i32_o64 | 32 | 64 | True | 529.838232 | 11.994073 | 4.269058 | 234.243722 | 248.893514 |  |
| i32_o128 | 32 | 128 | True | 1027.748058 | 13.008090 | 4.310770 | 231.977128 | 467.272201 |  |
| i32_o512 | 32 | 512 | True | 3616.627345 | 11.949081 | 4.081143 | 245.029399 | 1519.214222 |  |
| i128_o64 | 128 | 64 | True | 695.502609 | 12.008080 | 4.182742 | 239.077621 | 419.981785 |  |
| i128_o128 | 128 | 128 | True | 939.679151 | 11.830078 | 4.061743 | 246.199715 | 412.007697 |  |
| i128_o512 | 128 | 512 | True | 3521.619823 | 11.527075 | 3.983685 | 251.023843 | 1474.429549 |  |
| i512_o64 | 512 | 64 | True | 588.578778 | 11.429073 | 3.951676 | 253.057171 | 328.194104 |  |
| i512_o128 | 512 | 128 | True | 1074.724854 | 12.017077 | 4.225358 | 236.666353 | 526.087347 |  |
| i512_o512 | 512 | 512 | True | 3717.250596 | 12.387079 | 4.088231 | 244.604547 | 1615.777251 |  |
| i1024_o64 | 1024 | 64 | True | 590.432734 | 11.545073 | 3.956279 | 252.762765 | 329.642089 |  |
| i1024_o128 | 1024 | 128 | True | 951.066002 | 12.104077 | 4.095782 | 244.153636 | 418.797647 |  |
| i1024_o512 | 1024 | 512 | True | 3675.635155 | 12.000076 | 4.089744 | 244.514084 | 1573.775913 |  |

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
