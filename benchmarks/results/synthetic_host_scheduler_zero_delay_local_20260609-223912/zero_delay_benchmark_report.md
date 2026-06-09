# vLLM Zero-Delay Synthetic Host Scheduler Benchmark

## Conclusion
- This run sets synthetic prefill/prefix delay and decode delay to `0.0 ms`, so the measured latency approximates the local WSL CPU scheduler/control-plane lower bound.
- All matrix cases passed: `12/12`. Dummy output token validation and output length validation passed inside the wrapped latency benchmark.
- Zero-delay scheduler score: `4248.72` synthetic decode tokens/s, using the geometric mean of decode TPS across the 12-case matrix.
- Average TTFT/prefix latency: `0.337 ms`. Average TPOT/decode latency: `0.235 ms/token`.
- Compared with the previous `5 ms prefill / 1 ms decode` local WSL baseline, average TPOT improves from `1.283` to `0.235 ms/token`, showing the dummy sleep delay was the dominant part of the earlier decode path.

## Configuration
| item | value |
| --- | --- |
| result dir | `benchmarks/results/synthetic_host_scheduler_zero_delay_local_20260609-223912` |
| platform | local WSL / AMD 9950X3D host |
| benchmark command | `vllm bench synthetic` |
| dummy prefill/prefix delay | `0.0 ms` |
| dummy decode delay | `0.0 ms` |
| input token matrix | `32,128,512,1024` |
| output token matrix | `64,128,512` |
| repeats / warmup | `5 / 1` |
| dummy token id | `0` |
| max model len | `2048` |

## Benchmark Score
| metric | value |
| --- | ---: |
| geometric mean decode TPS | 4248.72 tok/s |
| harmonic mean decode TPS | 4247.66 tok/s |
| average TTFT / prefix latency | 0.3374 ms |
| average TPOT / decode latency | 0.2354 ms/token |
| average prefill step latency | 0.3374 ms |
| average decode step latency | 0.2354 ms |
| max TPOT | 0.2447 ms/token |
| min decode TPS | 4087.11 tok/s |

## Matrix Results
| case | TTFT ms | TPOT ms | decode TPS | host-adjusted E2E ms | prefill step ms | decode step ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| i32_o64 | 0.3235 | 0.2396 | 4173.1 | 8.3565 | 0.3235 | 0.2396 |
| i32_o128 | 0.3360 | 0.2390 | 4183.6 | 15.8776 | 0.3360 | 0.2390 |
| i32_o512 | 0.3364 | 0.2351 | 4253.9 | 60.1863 | 0.3364 | 0.2351 |
| i128_o64 | 0.3207 | 0.2438 | 4101.7 | 8.4432 | 0.3207 | 0.2438 |
| i128_o128 | 0.3174 | 0.2319 | 4312.5 | 15.5205 | 0.3174 | 0.2319 |
| i128_o512 | 0.3536 | 0.2447 | 4087.1 | 62.9605 | 0.3536 | 0.2447 |
| i512_o64 | 0.3345 | 0.2312 | 4324.8 | 8.1670 | 0.3345 | 0.2312 |
| i512_o128 | 0.3296 | 0.2302 | 4343.7 | 15.5083 | 0.3296 | 0.2302 |
| i512_o512 | 0.3431 | 0.2295 | 4356.9 | 59.5134 | 0.3431 | 0.2295 |
| i1024_o64 | 0.3465 | 0.2350 | 4255.3 | 8.3605 | 0.3465 | 0.2350 |
| i1024_o128 | 0.3487 | 0.2282 | 4381.6 | 15.5698 | 0.3487 | 0.2282 |
| i1024_o512 | 0.3588 | 0.2368 | 4223.2 | 60.8026 | 0.3588 | 0.2368 |

## Comparison With 5ms/1ms Baseline
| case | zero TPOT ms | baseline TPOT ms | TPOT speedup | zero TTFT ms | baseline TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| i32_o64 | 0.2396 | 1.2704 | 5.30x | 0.3235 | 5.4226 |
| i32_o128 | 0.2390 | 1.2779 | 5.35x | 0.3360 | 5.4781 |
| i32_o512 | 0.2351 | 1.2815 | 5.45x | 0.3364 | 5.4432 |
| i128_o64 | 0.2438 | 1.2961 | 5.32x | 0.3207 | 5.4627 |
| i128_o128 | 0.2319 | 1.2871 | 5.55x | 0.3174 | 5.4871 |
| i128_o512 | 0.2447 | 1.2788 | 5.23x | 0.3536 | 5.5029 |
| i512_o64 | 0.2312 | 1.2718 | 5.50x | 0.3345 | 5.4317 |
| i512_o128 | 0.2302 | 1.2724 | 5.53x | 0.3296 | 5.4196 |
| i512_o512 | 0.2295 | 1.2936 | 5.64x | 0.3431 | 5.4557 |
| i1024_o64 | 0.2350 | 1.2803 | 5.45x | 0.3465 | 5.4383 |
| i1024_o128 | 0.2282 | 1.2939 | 5.67x | 0.3487 | 5.4554 |
| i1024_o512 | 0.2368 | 1.2926 | 5.46x | 0.3588 | 5.4557 |

## Interpretation
- With dummy delays at zero, TPOT is stable around `0.23-0.24 ms/token`; this is the local host decode-loop lower bound under the current benchmark path.
- TTFT is stable around `0.32-0.36 ms`; because prefill compute is synthetic-zero, this mostly reflects scheduler/model-runner bookkeeping and dummy output construction.
- Input length has little effect in this synthetic-zero mode because no real prefill math is executed. Output length still affects total E2E time because decode iterates token by token through the host path.
- This score should not be read as real model throughput. It is specifically the CPU host scheduler/control-plane benchmark score when accelerator compute is reduced to near zero.

## Artifacts
- `matrix_results.json`: full structured benchmark output.
- `matrix_results.csv`: matrix table.
- `benchmark_score.json`: condensed score summary.
- `zero_delay_case_comparison.csv`: comparison against the earlier local `5ms/1ms` synthetic baseline.
- `raw_logs/`: one raw latency JSON/log per matrix cell.
