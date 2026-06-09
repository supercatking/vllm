# vLLM Synthetic Host Scheduler Four-Platform Benchmark Report
## Conclusion
- The synthetic host scheduler benchmark is now implemented and validated on the local WSL platform and on `dude` using the native `vllm bench synthetic` path.
- Local WSL is the fastest completed platform: average TTFT `5.455 ms`, average TPOT `1.283 ms/token`, average decode TPS about `779 tok/s`.
- `dude` completed the same 12-case matrix on RISC-V: average TTFT `11.983 ms`, average TPOT `4.108 ms/token`, average decode TPS about `244 tok/s`. Compared with local WSL, `dude` has roughly `3.2x` higher TPOT in this synthetic control-plane benchmark.
- `peter` could not be benchmarked because SSH remained unavailable for the 10-minute retry window after reboot.
- `orin` could not enter the vLLM benchmark path because the current Jetson environment has no complete vLLM install and its NVIDIA PyTorch build lacks `torch._C._distributed_c10d`. This is an environment compatibility blocker, not a benchmark result.

## Method
- Benchmark target: CPU-visible vLLM V1 scheduler/control-plane latency.
- Non-goal: real GPU/NPU kernel latency, model quality, or CPU model inference throughput.
- Synthetic device behavior: prefill sleeps `5 ms`, decode sleeps `1 ms`, output token is fixed to token id `0`.
- Matrix: input tokens `32,128,512,1024` by output tokens `64,128,512`; one repeat per case on remote boards to keep runtime bounded.
- Pass condition for completed platforms: each case must succeed, output token ids must be all dummy token `0`, and output length must match the requested length.

## Platform Summary
| platform | status | cases | avg TTFT ms | avg TPOT ms | avg decode TPS | note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| local_wsl_amd9950x3d | completed | 12/12 | 5.454 | 1.283 | 779.447 | native `vllm bench synthetic` |
| dude_riscv64_native | completed | 12/12 | 11.983 | 4.108 | 243.609 | native `vllm bench synthetic` |
| peter_riscv64 | blocked | 0/12 | - | - | - | SSH unavailable during retry window |
| orin_jetson_aarch64 | blocked | 0/12 | - | - | - | incomplete vLLM install and PyTorch distributed module missing |

## Completed Matrix: Local WSL vs Dude
| case | local TTFT | local TPOT | local TPS | dude TTFT | dude TPOT | dude TPS | dude TPOT slowdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| i32_o64 | 5.423 | 1.270 | 787.184 | 11.994 | 4.269 | 234.244 | 3.361x |
| i32_o128 | 5.478 | 1.278 | 782.504 | 13.008 | 4.311 | 231.977 | 3.373x |
| i32_o512 | 5.443 | 1.281 | 780.346 | 11.949 | 4.081 | 245.029 | 3.185x |
| i128_o64 | 5.463 | 1.296 | 771.559 | 12.008 | 4.183 | 239.078 | 3.227x |
| i128_o128 | 5.487 | 1.287 | 776.947 | 11.830 | 4.062 | 246.200 | 3.156x |
| i128_o512 | 5.503 | 1.279 | 781.995 | 11.527 | 3.984 | 251.024 | 3.115x |
| i512_o64 | 5.432 | 1.272 | 786.293 | 11.429 | 3.952 | 253.057 | 3.107x |
| i512_o128 | 5.420 | 1.272 | 785.916 | 12.017 | 4.225 | 236.666 | 3.321x |
| i512_o512 | 5.456 | 1.294 | 773.035 | 12.387 | 4.088 | 244.605 | 3.160x |
| i1024_o64 | 5.438 | 1.280 | 781.088 | 11.545 | 3.956 | 252.763 | 3.090x |
| i1024_o128 | 5.455 | 1.294 | 772.876 | 12.104 | 4.096 | 244.154 | 3.166x |
| i1024_o512 | 5.456 | 1.293 | 773.619 | 12.000 | 4.090 | 244.514 | 3.164x |

## Findings
- Local WSL overhead is very stable across input lengths: TTFT stays around `5.42-5.50 ms`, and TPOT stays around `1.27-1.30 ms/token`. This indicates the dummy path is dominated by the configured 5 ms/1 ms synthetic delays plus a small and stable host overhead.
- `dude` also shows stable behavior across input lengths, but with a larger host/control-plane cost: TTFT is around `11.4-13.0 ms`, and TPOT is around `3.98-4.31 ms/token`.
- Because this benchmark forces the same synthetic device delay, the local-vs-dude gap mainly represents Python/vLLM scheduler/control-plane overhead on the host CPU, not model math.
- Output length drives host-adjusted E2E latency more strongly than input length, as expected for scheduler/decode loops. Input length has limited effect in this synthetic mode because prefill is represented by a fixed delay.

## Blockers
- `peter`: SSH alias `peter` did not accept a login during the retry window. Artifact: `benchmarks/results/_remote_retry_peter/blocker_ssh_unavailable_20260609_095412.log`.
- `orin`: environment probe reports PyTorch `2.5.0a0+872d972e41.nv24.08` at `/usr/local/lib/python3.10/dist-packages/torch`, and `import torch._C._distributed_c10d` fails. Artifact: `benchmarks/results/_remote_retry_orin/orin_native_fixed_latest/env_probe.txt`.

## Artifacts
- Local WSL results: `benchmarks/results/synthetic_host_scheduler_20260609-000821/local_wsl/`.
- Dude native results: `benchmarks/results/_remote_retry_dude/dude_native_matrix_20260609_093542/`.
- Peter blocker logs: `benchmarks/results/_remote_retry_peter/`.
- Orin blocker logs: `benchmarks/results/_remote_retry_orin/orin_native_fixed_latest/`.
