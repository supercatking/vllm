# orin retry blocker

Remote sandbox: `/home/nvidia/vllm_synthetic_host_benchmark/orin_20260609_083900`
Local copied results/logs: `/home/zyz/vLLM_deploy/vllm/benchmarks/results/_remote_retry_orin/orin_20260609_083900`

Smoke did not pass, so the 4x3 matrix was not run.

Final smoke command failed during vLLM CLI import before benchmark execution:

```text
ModuleNotFoundError: No module named 'torch._C._distributed_c10d'; 'torch._C' is not a package
```

Earlier blockers/workarounds observed:

- `python3 -m venv` failed because remote Python lacks `ensurepip` / `python3.10-venv`; sudo was not used.
- Created a sandboxed virtualenv via `python3 -m pip install --target ... virtualenv` under the run directory.
- Global remote Python had torch `2.5.0a0+872d972e41.nv24.08`, CUDA `12.6`, `torch.cuda.is_available() == True`, but no `torch.library.infer_schema`.
- Global remote Python had no `transformers` and no importable `vllm`.
- Copied source tree contained incompatible local `.so` artifacts, removed only inside the remote sandbox copy.
- Added sandbox-only shims for `torch.library.infer_schema` and `torch._inductor.custom_graph_pass.CustomGraphPass` to get past earlier import-time failures.
- Forced sandbox metadata to `vllm 0.0.0+cpu` to avoid CUDA platform importing missing aarch64 vLLM extension modules, but import then failed because Jetson torch lacks distributed c10d internals required by this vLLM checkout.

No benchmark metrics were produced.