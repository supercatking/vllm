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
| i32_o64 | 32 | 64 | True | 116.708488 | 1.261786 | 0.762718 | 1311.099833 | 67.395442 |  |
| i32_o128 | 32 | 128 | True | 221.773828 | 1.249591 | 0.774938 | 1290.426114 | 122.107134 |  |
| i32_o512 | 32 | 512 | True | 868.001170 | 1.281796 | 0.769078 | 1300.258634 | 473.720638 |  |
| i128_o64 | 128 | 64 | True | 113.686278 | 1.288598 | 0.761501 | 1313.196284 | 64.423130 |  |
| i128_o128 | 128 | 128 | True | 221.257024 | 1.280001 | 0.767220 | 1303.407657 | 122.540124 |  |
| i128_o512 | 128 | 512 | True | 866.919142 | 1.306603 | 0.766760 | 1304.188717 | 473.798077 |  |
| i512_o64 | 512 | 64 | True | 119.080281 | 1.358805 | 0.768892 | 1300.572756 | 69.281279 |  |
| i512_o128 | 512 | 128 | True | 221.723728 | 1.348407 | 0.766812 | 1304.100976 | 122.990224 |  |
| i512_o512 | 512 | 512 | True | 863.912530 | 1.389008 | 0.761503 | 1313.192897 | 473.395617 |  |
| i1024_o64 | 1024 | 64 | True | 116.830772 | 1.489010 | 0.789745 | 1266.231689 | 65.587835 |  |
| i1024_o128 | 1024 | 128 | True | 226.526017 | 1.487010 | 0.766157 | 1305.216139 | 127.737114 |  |
| i1024_o512 | 1024 | 512 | True | 878.625849 | 1.514612 | 0.775154 | 1290.065896 | 481.007440 |  |

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
