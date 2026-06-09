# vLLM Synthetic Host Scheduler Benchmark

This report measures CPU-visible vLLM host/scheduler overhead with synthetic GPU execution. It does not measure model quality, CPU model execution, or real accelerator kernel performance.

## Configuration

- model: `/home/ubuntu/vllm/.p550_models/qwen2.5-0.5b-instruct`
- backend: `sleep`
- prefill delay: `0.0 ms`
- decode delay: `0.0 ms`
- dummy output token: `0`
- batch size: `1`
- repeats/warmup: `5/1`

## Environment

- hostname: `ubuntu`
- machine: `riscv64`
- kernel: `Linux ubuntu 6.6.92-2025-eic7700 #10 SMP Sun Nov  2 20:08:44 UTC 2025 riscv64 riscv64 riscv64 GNU/Linux`
- vLLM commit: `fatal: not a git repository (or any of the parent directories): .git`

## Results

| case_id | input_tokens_target | output_tokens_target | success | avg_e2e_ms | ttft_avg_ms | tpot_ms | decode_tps | host_adjusted_e2e_ms | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| i32_o64 | 32 | 64 | True | 114.893947 | 1.253993 | 0.785849 | 1272.508314 | 64.131437 |  |
| i32_o128 | 32 | 128 | True | 227.638521 | 1.273793 | 0.777950 | 1285.429722 | 127.565084 |  |
| i32_o512 | 32 | 512 | True | 873.616946 | 1.298193 | 0.773013 | 1293.639418 | 477.309145 |  |
| i128_o64 | 128 | 64 | True | 114.140972 | 1.256993 | 0.780177 | 1281.760964 | 63.732848 |  |
| i128_o128 | 128 | 128 | True | 233.823522 | 1.332393 | 0.767130 | 1303.560563 | 135.065660 |  |
| i128_o512 | 128 | 512 | True | 869.990290 | 1.314114 | 0.765930 | 1305.602196 | 477.285903 |  |
| i512_o64 | 512 | 64 | True | 118.766229 | 1.376102 | 0.797434 | 1254.022436 | 67.151791 |  |
| i512_o128 | 512 | 128 | True | 224.195383 | 1.375689 | 0.774076 | 1291.862131 | 124.511990 |  |
| i512_o512 | 512 | 512 | True | 870.907896 | 1.413278 | 0.766191 | 1305.157502 | 477.970991 |  |
| i1024_o64 | 1024 | 64 | True | 114.711678 | 1.459070 | 0.768141 | 1301.843532 | 64.859696 |  |
| i1024_o128 | 1024 | 128 | True | 228.432877 | 1.497062 | 0.766485 | 1304.656397 | 129.592177 |  |
| i1024_o512 | 1024 | 512 | True | 871.893798 | 1.504854 | 0.765444 | 1306.431296 | 479.247059 |  |

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
