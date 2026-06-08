# vLLM Synthetic Host Scheduler Four-Platform Summary

## Conclusion

- The local WSL platform completed the full 12-case synthetic matrix successfully.
- `peter`, `dude`, and `orin` did not produce performance data. They are recorded as blockers, not failed benchmark measurements.
- This run evaluates the synthetic/dummy accelerator path for CPU-visible vLLM host scheduler/control-plane overhead. It is deliberately separate from the earlier CPU backend qwen0.5B inference deployments on the RISC-V boards.

## Platform Status

| Platform | Status | Result directory | Reason / next step |
| --- | --- | --- | --- |
| local_wsl_amd9950x3d_rtx | pass | `local_wsl` | 12 cases passed |
| peter_riscv | blocked_before_smoke | `peter/peter_20260608_161432` | Existing P550 vLLM environment fails CLI import: ModuleNotFoundError: No module named 'model_hosting_container_standards'. No synthetic benchmark was run to avoid contaminating the CPU backend environment.<br>`cd /home/ubuntu/vllm_p550dev_validation && source /home/ubuntu/vllm/.venv-p550/bin/activate && python -m pip install model-hosting-container-standards` |
| dude_riscv | blocked_ssh_access | `dude` | Non-interactive SSH failed: Permission denied (publickey,password).<br>`ssh-copy-id -i /home/zyz/.ssh/id_ed25519.pub dude` |
| orin_jetson_aarch64 | blocked_before_smoke | `orin/orin_20260609_001328` | Remote Python/vLLM deps are incomplete/incompatible: ImportError cannot import name 'infer_schema' from torch.library; transformers is not installed. Smoke exited before benchmark initialization. |

## Local WSL Matrix

| Case | Input tokens | Output tokens | E2E ms | TTFT ms | TPOT ms | Decode TPS | Host-adjusted E2E ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| i32_o64 | 32 | 64 | 94.621 | 5.423 | 1.270 | 787.184 | 9.166 |
| i32_o128 | 32 | 128 | 186.003 | 5.478 | 1.278 | 782.504 | 18.226 |
| i32_o512 | 32 | 512 | 730.017 | 5.443 | 1.281 | 780.346 | 69.736 |
| i128_o64 | 128 | 64 | 97.513 | 5.463 | 1.296 | 771.559 | 10.398 |
| i128_o128 | 128 | 128 | 188.002 | 5.487 | 1.287 | 776.947 | 19.054 |
| i128_o512 | 128 | 512 | 726.845 | 5.503 | 1.279 | 781.995 | 67.886 |
| i512_o64 | 512 | 64 | 94.981 | 5.432 | 1.272 | 786.293 | 9.427 |
| i512_o128 | 512 | 128 | 184.616 | 5.420 | 1.272 | 785.916 | 17.601 |
| i512_o512 | 512 | 512 | 736.455 | 5.456 | 1.294 | 773.035 | 69.969 |
| i1024_o64 | 1024 | 64 | 95.844 | 5.438 | 1.280 | 781.088 | 9.749 |
| i1024_o128 | 1024 | 128 | 189.025 | 5.455 | 1.294 | 772.876 | 19.248 |
| i1024_o512 | 1024 | 512 | 739.057 | 5.456 | 1.293 | 773.619 | 73.069 |

## Local Interpretation

- Average TPOT across the matrix was `1.283 ms/token`.
- Average host-adjusted E2E latency was `32.794 ms`.
- TTFT stayed close to the configured 5 ms synthetic prefill delay, which indicates the local scheduler path is adding sub-millisecond TTFT overhead in this smoke-sized setup.
- Longer output cases show host-adjusted E2E growth, so per-token host/output handling remains visible even when accelerator compute is replaced by dummy sleep.

## Blocker Details

### peter

The existing RISC-V P550 environment was left untouched because it contains CPU backend work. `vllm bench --help` fails before synthetic benchmark registration due to `model_hosting_container_standards` missing.

### dude

Non-interactive SSH still fails. Install the WSL public key before running remote automation.

### orin

The isolated source tree reached import probing, but the Python environment lacks compatible vLLM dependencies: `torch.library.infer_schema` is unavailable and `transformers` is missing.
