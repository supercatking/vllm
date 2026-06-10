# p550dummy Zero-Delay Synthetic Host Scheduler Benchmark Rerun

## Conclusion

- Local WSL completed two valid zero-delay dummy rounds: 24/24 cases passed. Combined avg TTFT `0.363 ms`, avg TPOT `0.250 ms/token`, geomean decode TPS `3993.1 tok/s`.
- peter completed two valid p550dummy zero-delay rounds: 24/24 cases passed. Combined avg TTFT `1.368 ms`, avg TPOT `0.769 ms/token`, geomean decode TPS `1299.7 tok/s`.
- dude did not produce a score because `192.168.1.57:22` was unreachable from WSL/Windows. This is recorded as a network/access blocker, not a Python dependency failure.

The peter score is stable across the two fresh reruns: round avg TPOT differed by only `0.000 ms/token`, and decode TPS stayed around `1.30k tok/s`. The measured prefill latency is the latency for the full scheduled prefill chunk, so an approximate per-input-token prefill cost is `prefill_latency / input_tokens`; the matrix below includes that derived value in microseconds/token.

## Method

- Benchmark mode: vLLM V1 synthetic dummy execution.
- Device delay configuration: prefill `0 ms`, decode `0 ms`.
- Dummy token: `0`; every recorded prefill/decode step had `dummy_token_id=0`, and decode step counts matched `(output_len - 1) * repeats`.
- Matrix: input tokens `32,128,512,1024`; output tokens `64,128,512`; warmup `1`; repeats `5`; batch size `1`; concurrency `1`.
- Local WSL used the benchmark branch synthetic path because p550dummy/p550dev is RISC-V CPU-targeted and fails on local WSL GPU/flash-attn imports. peter used the p550dummy source path directly.

## Round Scores

| platform | round | pass | avg TTFT ms | avg TPOT ms/token | geomean decode TPS | avg E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local_wsl | 1 | 12/12 | 0.359 | 0.250 | 4004.4 | 87.646 |
| local_wsl | 2 | 12/12 | 0.367 | 0.251 | 3981.8 | 88.502 |
| peter | 1 | 12/12 | 1.355 | 0.769 | 1300.1 | 402.920 |
| peter | 2 | 12/12 | 1.381 | 0.770 | 1299.3 | 410.768 |

## Combined Platform Scores

| platform | pass | avg TTFT ms | avg TPOT ms/token | geomean decode TPS | avg host-adjusted E2E ms | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| local_wsl | 24/24 | 0.363 | 0.250 | 3993.1 | 29.526 | benchmark branch synthetic path |
| peter | 24/24 | 1.368 | 0.769 | 1299.7 | 226.200 | p550dummy branch on RISC-V |
| dude | 0/24 | n/a | n/a | n/a | n/a | network blocker: `192.168.1.57:22` unreachable |

## Case Matrix Averages

| case | in | out | local TTFT ms | local prefill us/input tok | local TPOT ms | local TPS | peter TTFT ms | peter prefill us/input tok | peter TPOT ms | peter TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| i32_o64 | 32 | 64 | 0.366 | 11.452 | 0.247 | 4043.4 | 1.249 | 39.025 | 0.764 | 1309.7 |
| i32_o128 | 32 | 128 | 0.353 | 11.024 | 0.256 | 3905.6 | 1.279 | 39.963 | 0.771 | 1297.6 |
| i32_o512 | 32 | 512 | 0.353 | 11.034 | 0.246 | 4066.9 | 1.314 | 41.071 | 0.763 | 1310.4 |
| i128_o64 | 128 | 64 | 0.338 | 2.644 | 0.251 | 3992.7 | 1.286 | 10.045 | 0.767 | 1303.1 |
| i128_o128 | 128 | 128 | 0.362 | 2.826 | 0.253 | 3955.2 | 1.287 | 10.056 | 0.765 | 1307.1 |
| i128_o512 | 128 | 512 | 0.349 | 2.729 | 0.249 | 4019.7 | 1.312 | 10.253 | 0.766 | 1305.1 |
| i512_o64 | 512 | 64 | 0.360 | 0.704 | 0.251 | 3990.0 | 1.372 | 2.680 | 0.777 | 1287.1 |
| i512_o128 | 512 | 128 | 0.352 | 0.688 | 0.249 | 4018.7 | 1.377 | 2.689 | 0.769 | 1299.6 |
| i512_o512 | 512 | 512 | 0.379 | 0.740 | 0.250 | 4006.5 | 1.395 | 2.725 | 0.761 | 1313.5 |
| i1024_o64 | 1024 | 64 | 0.365 | 0.356 | 0.254 | 3938.9 | 1.486 | 1.451 | 0.783 | 1277.6 |
| i1024_o128 | 1024 | 128 | 0.401 | 0.391 | 0.254 | 3932.7 | 1.540 | 1.504 | 0.774 | 1292.7 |
| i1024_o512 | 1024 | 512 | 0.373 | 0.364 | 0.247 | 4050.6 | 1.514 | 1.479 | 0.773 | 1293.5 |

## Analysis

- Zero-delay mode isolates host-side scheduler/control-plane latency. The dummy backend removes intentional device delay, so remaining TTFT/TPOT reflects vLLM request scheduling, token output plumbing, process/runtime overhead, and CPU-side control flow.
- Local WSL is roughly `3.1x` faster than peter on TPOT in this matrix (`0.250` vs `0.769` ms/token), which is expected given AMD desktop CPU versus RISC-V board CPU. The relevant peter benchmark score for decode capacity is `1299.7 tok/s`.
- peter TTFT is also higher than local WSL (`1.368` vs `0.363` ms), but still low in absolute terms. Since prefill is measured as one scheduled chunk, the per-token prefill value falls as input length grows; it is a derived normalization, not an independently measured token-by-token prefill loop.
- The raw wall time on peter is dominated by repeated Python/vLLM engine startup per matrix case. That startup time is outside TTFT/TPOT and should not be interpreted as per-request scheduler latency.
- dude has no valid data in this rerun. Any comparison table must exclude dude until the board is reachable over SSH.

## Artifacts

- `summary.json`: structured aggregate scores and validation status.
- `case_matrix_comparison.csv`: per-case local/peter comparison table.
- `local_wsl_benchmark_branch_zero_2rounds/`: local two-round raw results and logs.
- `peter_p550dummy_zero_2rounds/`: peter two-round raw results and logs.
- `dude_blocked_network_unreachable/`: dude network blocker evidence.
