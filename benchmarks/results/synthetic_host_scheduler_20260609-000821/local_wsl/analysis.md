# vLLM Synthetic Host Scheduler Benchmark

This report measures CPU-visible vLLM host/scheduler overhead with synthetic GPU execution. It does not measure model quality, CPU model execution, or real accelerator kernel performance.

## Configuration

- model: `/home/zyz/vLLM_deploy/localModels/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- backend: `sleep`
- prefill delay: `5.0 ms`
- decode delay: `1.0 ms`
- dummy output token: `0`
- batch size: `1`
- repeats/warmup: `5/1`

## Environment

- hostname: `amd9950`
- machine: `x86_64`
- kernel: `Linux amd9950 6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun  5 18:30:46 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux`
- vLLM commit: `8b68754650487e7b8010e340d1032692db8f52d6`

## Results

| case_id | input_tokens_target | output_tokens_target | success | avg_e2e_ms | ttft_avg_ms | tpot_ms | decode_tps | host_adjusted_e2e_ms | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| i32_o64 | 32 | 64 | True | 94.620521 | 5.422621 | 1.270352 | 787.183678 | 9.165752 |  |
| i32_o128 | 32 | 128 | True | 186.003199 | 5.478074 | 1.277949 | 782.503755 | 18.225585 |  |
| i32_o512 | 32 | 512 | True | 730.017341 | 5.443216 | 1.281483 | 780.346037 | 69.736431 |  |
| i128_o64 | 128 | 64 | True | 97.513320 | 5.462720 | 1.296078 | 771.558521 | 10.397688 |  |
| i128_o128 | 128 | 128 | True | 188.001572 | 5.487085 | 1.287089 | 776.947123 | 19.054200 |  |
| i128_o512 | 128 | 512 | True | 726.845481 | 5.502936 | 1.278780 | 781.995417 | 67.886030 |  |
| i512_o64 | 512 | 64 | True | 94.981365 | 5.431673 | 1.271790 | 786.293091 | 9.426897 |  |
| i512_o128 | 512 | 128 | True | 184.615737 | 5.419607 | 1.272401 | 785.915651 | 17.601181 |  |
| i512_o512 | 512 | 512 | True | 736.455281 | 5.455651 | 1.293602 | 773.035190 | 69.968921 |  |
| i1024_o64 | 1024 | 64 | True | 95.844230 | 5.438306 | 1.280266 | 781.087824 | 9.749182 |  |
| i1024_o128 | 1024 | 128 | True | 189.025035 | 5.455416 | 1.293868 | 772.876106 | 19.248327 |  |
| i1024_o512 | 1024 | 512 | True | 739.056722 | 5.455651 | 1.292626 | 773.618809 | 73.069045 |  |

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
