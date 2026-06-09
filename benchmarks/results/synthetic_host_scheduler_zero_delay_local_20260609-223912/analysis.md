# vLLM Synthetic Host Scheduler Benchmark

This report measures CPU-visible vLLM host/scheduler overhead with synthetic GPU execution. It does not measure model quality, CPU model execution, or real accelerator kernel performance.

## Configuration

- model: `/home/zyz/vLLM_deploy/localModels/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- backend: `sleep`
- prefill delay: `0.0 ms`
- decode delay: `0.0 ms`
- dummy output token: `0`
- batch size: `1`
- repeats/warmup: `5/1`

## Environment

- hostname: `amd9950`
- machine: `x86_64`
- kernel: `Linux amd9950 6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun  5 18:30:46 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux`
- vLLM commit: `4761c996d7d9f534f6a7eed449af206edd8ca684`

## Results

| case_id | input_tokens_target | output_tokens_target | success | avg_e2e_ms | ttft_avg_ms | tpot_ms | decode_tps | host_adjusted_e2e_ms | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| i32_o64 | 32 | 64 | True | 23.776809 | 0.323450 | 0.239633 | 4173.050046 | 8.356487 |  |
| i32_o128 | 32 | 128 | True | 46.570216 | 0.336033 | 0.239028 | 4183.611886 | 15.877639 |  |
| i32_o512 | 32 | 512 | True | 180.648501 | 0.336413 | 0.235080 | 4253.873445 | 60.186276 |  |
| i128_o64 | 128 | 64 | True | 24.123534 | 0.320709 | 0.243804 | 4101.654155 | 8.443168 |  |
| i128_o128 | 128 | 128 | True | 45.287055 | 0.317388 | 0.231883 | 4312.512717 | 15.520478 |  |
| i128_o512 | 128 | 512 | True | 188.341465 | 0.353596 | 0.244672 | 4087.105015 | 62.960497 |  |
| i512_o64 | 512 | 64 | True | 23.068554 | 0.334509 | 0.231222 | 4324.839380 | 8.167030 |  |
| i512_o128 | 512 | 128 | True | 45.075680 | 0.329591 | 0.230219 | 4343.692509 | 15.508292 |  |
| i512_o512 | 512 | 512 | True | 177.142742 | 0.343123 | 0.229523 | 4356.862663 | 59.513388 |  |
| i1024_o64 | 1024 | 64 | True | 23.512247 | 0.346545 | 0.235003 | 4255.258509 | 8.360491 |  |
| i1024_o128 | 1024 | 128 | True | 44.903507 | 0.348651 | 0.228229 | 4381.566907 | 15.569792 |  |
| i1024_o512 | 1024 | 512 | True | 182.158913 | 0.358757 | 0.236786 | 4223.224154 | 60.802553 |  |

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
