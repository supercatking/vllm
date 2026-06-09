# Peter p550dummy Zero-Delay Synthetic Benchmark Report

## Conclusion
- `p550dummy` was forked from `p550dev` and received the dummy GPU / synthetic host scheduler benchmark changes from the `benchmark` branch.
- The benchmark was run on `peter` using the existing P550/RISC-V vLLM environment and local Qwen2.5-0.5B model path.
- All matrix cases passed: `12/12`.
- Peter benchmark score: `1292.11` synthetic decode tok/s, defined as geometric mean decode TPS across the 12-case matrix.
- Average TTFT/prefill latency: `1.363 ms`; average TPOT/decode latency: `0.774 ms/token`.

## Configuration
| item | value |
| --- | --- |
| board | `peter` |
| branch under test | `p550dummy` |
| base branch | `p550dev` |
| model | `/home/ubuntu/vllm/.p550_models/qwen2.5-0.5b-instruct` |
| dummy prefill latency | `0.0 ms` |
| dummy decode latency | `0.0 ms` |
| input token matrix | `32,128,512,1024` |
| output token matrix | `64,128,512` |
| repeats / warmup | `5 / 1` |
| target device | `cpu` with dummy execution path |

## Benchmark Score
| metric | value |
| --- | ---: |
| geometric mean decode TPS | 1292.11 tok/s |
| harmonic mean decode TPS | 1292.01 tok/s |
| average TTFT / prefill latency | 1.3630 ms |
| average TPOT / decode latency | 0.7740 ms/token |
| average host-adjusted E2E | 224.0353 ms |
| max TPOT | 0.7974 ms/token |
| min decode TPS | 1254.02 tok/s |

## Matrix Results
| case | TTFT ms | TPOT ms | decode TPS | host-adjusted E2E ms | prefill step ms | decode step ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| i32_o64 | 1.2540 | 0.7858 | 1272.5 | 64.1314 | 1.2540 | 0.7858 |
| i32_o128 | 1.2738 | 0.7779 | 1285.4 | 127.5651 | 1.2738 | 0.7779 |
| i32_o512 | 1.2982 | 0.7730 | 1293.6 | 477.3091 | 1.2982 | 0.7730 |
| i128_o64 | 1.2570 | 0.7802 | 1281.8 | 63.7328 | 1.2570 | 0.7802 |
| i128_o128 | 1.3324 | 0.7671 | 1303.6 | 135.0657 | 1.3324 | 0.7671 |
| i128_o512 | 1.3141 | 0.7659 | 1305.6 | 477.2859 | 1.3141 | 0.7659 |
| i512_o64 | 1.3761 | 0.7974 | 1254.0 | 67.1518 | 1.3761 | 0.7974 |
| i512_o128 | 1.3757 | 0.7741 | 1291.9 | 124.5120 | 1.3757 | 0.7741 |
| i512_o512 | 1.4133 | 0.7662 | 1305.2 | 477.9710 | 1.4133 | 0.7662 |
| i1024_o64 | 1.4591 | 0.7681 | 1301.8 | 64.8597 | 1.4591 | 0.7681 |
| i1024_o128 | 1.4971 | 0.7665 | 1304.7 | 129.5922 | 1.4971 | 0.7665 |
| i1024_o512 | 1.5049 | 0.7654 | 1306.4 | 479.2471 | 1.5049 | 0.7654 |

## Comparison With Local WSL Zero-Delay Result
| case | peter TPOT ms | local WSL TPOT ms | peter slowdown | peter TPS | local WSL TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| i32_o64 | 0.7858 | 0.2396 | 3.28x | 1272.5 | 4173.1 |
| i32_o128 | 0.7779 | 0.2390 | 3.25x | 1285.4 | 4183.6 |
| i32_o512 | 0.7730 | 0.2351 | 3.29x | 1293.6 | 4253.9 |
| i128_o64 | 0.7802 | 0.2438 | 3.20x | 1281.8 | 4101.7 |
| i128_o128 | 0.7671 | 0.2319 | 3.31x | 1303.6 | 4312.5 |
| i128_o512 | 0.7659 | 0.2447 | 3.13x | 1305.6 | 4087.1 |
| i512_o64 | 0.7974 | 0.2312 | 3.45x | 1254.0 | 4324.8 |
| i512_o128 | 0.7741 | 0.2302 | 3.36x | 1291.9 | 4343.7 |
| i512_o512 | 0.7662 | 0.2295 | 3.34x | 1305.2 | 4356.9 |
| i1024_o64 | 0.7681 | 0.2350 | 3.27x | 1301.8 | 4255.3 |
| i1024_o128 | 0.7665 | 0.2282 | 3.36x | 1304.7 | 4381.6 |
| i1024_o512 | 0.7654 | 0.2368 | 3.23x | 1306.4 | 4223.2 |

## Interpretation
- With both synthetic prefill and decode latency set to zero, this score reflects the CPU-visible vLLM scheduler/control-plane path on peter, not real model math.
- Input length has limited effect because prefill compute is skipped; output length still increases total E2E latency because the host decode loop iterates token by token.
- The result verifies that the p550dev CPU backend environment can run the migrated dummy backend benchmark path without loading real model weights for compute.

## Artifacts
- `matrix_results.json`: complete structured benchmark output.
- `matrix_results.csv`: matrix summary.
- `benchmark_score.json`: condensed score.
- `peter_zero_case_comparison.csv`: per-case comparison with local WSL zero-delay result when available.
- `raw_logs/`: raw wrapped latency JSON/log per matrix cell.
