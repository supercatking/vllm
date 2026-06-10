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
- vLLM commit: `44cb16f45878086c892cf7162f5760f4cba7d9d0`

## Results

| case_id | input_tokens_target | output_tokens_target | success | avg_e2e_ms | ttft_avg_ms | tpot_ms | decode_tps | host_adjusted_e2e_ms | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| i32_o64 | 32 | 64 | True | 24.770742 | 0.342733 | 0.248631 | 4022.017664 | 8.764230 |  |
| i32_o128 | 32 | 128 | True | 50.508272 | 0.353406 | 0.260045 | 3845.494203 | 17.129202 |  |
| i32_o512 | 32 | 512 | True | 188.560304 | 0.358233 | 0.246188 | 4061.932589 | 62.399885 |  |
| i128_o64 | 128 | 64 | True | 25.442249 | 0.350363 | 0.255205 | 3918.416711 | 9.013964 |  |
| i128_o128 | 128 | 128 | True | 48.732170 | 0.336051 | 0.250344 | 3994.510978 | 16.602490 |  |
| i128_o512 | 128 | 512 | True | 189.940173 | 0.361266 | 0.247736 | 4036.553000 | 62.985746 |  |
| i512_o64 | 512 | 64 | True | 24.533598 | 0.358963 | 0.245485 | 4073.568290 | 8.709078 |  |
| i512_o128 | 512 | 128 | True | 46.908318 | 0.336413 | 0.240966 | 4149.961730 | 15.969214 |  |
| i512_o512 | 512 | 512 | True | 186.753295 | 0.401875 | 0.243904 | 4099.974504 | 61.716499 |  |
| i1024_o64 | 1024 | 64 | True | 24.945791 | 0.351179 | 0.249981 | 4000.300421 | 8.845795 |  |
| i1024_o128 | 1024 | 128 | True | 51.385312 | 0.380469 | 0.262368 | 3811.438447 | 17.684090 |  |
| i1024_o512 | 1024 | 512 | True | 189.272775 | 0.372950 | 0.246714 | 4053.272361 | 62.828850 |  |

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
