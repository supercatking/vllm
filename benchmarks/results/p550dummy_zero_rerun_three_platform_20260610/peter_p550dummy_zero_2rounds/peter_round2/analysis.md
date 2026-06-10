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
| i32_o64 | 32 | 64 | True | 116.267531 | 1.235810 | 0.764390 | 1308.232280 | 66.875135 |  |
| i32_o128 | 32 | 128 | True | 232.415518 | 1.308011 | 0.766450 | 1304.715850 | 133.768305 |  |
| i32_o512 | 32 | 512 | True | 896.505195 | 1.346777 | 0.757167 | 1320.712637 | 508.246090 |  |
| i128_o64 | 128 | 64 | True | 114.798418 | 1.282984 | 0.773308 | 1293.145544 | 64.797020 |  |
| i128_o128 | 128 | 128 | True | 238.777808 | 1.294389 | 0.762838 | 1310.894842 | 140.603032 |  |
| i128_o512 | 128 | 512 | True | 882.032694 | 1.318193 | 0.765746 | 1305.915720 | 489.418193 |  |
| i512_o64 | 512 | 64 | True | 122.480036 | 1.385996 | 0.785061 | 1273.786241 | 71.635192 |  |
| i512_o128 | 512 | 128 | True | 229.655944 | 1.404799 | 0.772084 | 1295.195558 | 130.196454 |  |
| i512_o512 | 512 | 512 | True | 873.097542 | 1.401200 | 0.761200 | 1313.714669 | 482.722988 |  |
| i1024_o64 | 1024 | 64 | True | 116.362184 | 1.483002 | 0.775769 | 1289.042715 | 66.005704 |  |
| i1024_o128 | 1024 | 128 | True | 228.954775 | 1.592404 | 0.781078 | 1280.282572 | 128.165520 |  |
| i1024_o512 | 1024 | 512 | True | 877.874073 | 1.513405 | 0.771012 | 1296.996813 | 482.373578 |  |

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
