# P550 NPU Framework Validation Notes

Validation date: 2026-06-14

Branch base: `p550dev`

## Local WSL Validation

Commands:

```bash
PYTHONPATH=. /home/zyz/vLLM_deploy/.venv/bin/python \
  -m pytest --confcutdir=tests/p550 tests/p550/test_npu_framework.py -q

/home/zyz/vLLM_deploy/.venv/bin/python -m compileall -q \
  vllm/p550 \
  vllm/platforms/p550_npu.py \
  vllm/v1/worker/p550_npu_worker.py \
  tests/p550/test_npu_framework.py

PYTHONPATH=. /home/zyz/vLLM_deploy/.venv/bin/python \
  -m vllm.p550.npu.cli run \
  --op rms_norm \
  --shape 2x8 \
  --backend compare \
  --report /tmp/p550_npu_cli.jsonl
```

Result:

- Unit tests: `5 passed`
- CLI compare: `ok=true`
- `rms_norm` output shape: `[2, 8]`
- Max absolute error: `0.0`
- Max relative error: `0.0`

Note: the repository-level `tests/conftest.py` requires optional dependencies
such as `tblib`, so the focused P550 tests were run with
`--confcutdir=tests/p550`.

## Peter Board Validation

Host:

- SSH alias: `peter`
- OS: Ubuntu 24.04.3 LTS
- Kernel: `6.6.92-2025-eic7700`
- Architecture: `riscv64`
- Python used: `/home/ubuntu/vllm/.venv-p550/bin/python`

NPU runtime probe:

- `/dev/npu0` exists.
- `eic7700_npu` and `eic7700_dsp` kernel modules are loaded.
- `/usr/include/essdk/es_npu_interface.h` exists.
- `/usr/lib/libedla_runtime.so` exists.
- `/opt/eswin/bin/es_run_model` exists.
- Current NPU devfreq observed: `1500000000`.

Smoke command:

```bash
cd /tmp/vllm_p550_npu_probe
PYTHONPATH=. /home/ubuntu/vllm/.venv-p550/bin/python \
  -m vllm.p550.npu.cli run \
  --op matmul \
  --shape 2x4 \
  --rhs-shape 4x3 \
  --backend compare \
  --report /tmp/p550_npu_probe.jsonl
```

Result:

- CLI compare: `ok=true`
- `matmul` output shape: `[2, 3]`
- Max absolute error: `0.0`
- Max relative error: `0.0`

The smoke test intentionally used CPU fallback runtime because no real NPU
operator runtime class was provided yet. This validates the golden-comparison
framework and peter's Python/Torch runtime without modifying peter's existing
CPU vLLM deployment.

## Dude Board Status

SSH alias: `dude`

Result:

```text
ssh: connect to host 192.168.1.57 port 22: No route to host
```

The dude board was not reachable during this validation pass, so no runtime
test was executed there.

## EIC7700 NPU Development Conclusion

Peter has the device, driver, headers, runtime library, and sample model runner
needed for validation with precompiled EIC7700 `.model` artifacts.

The board image did not expose the offline compiler toolchain needed to compile
arbitrary ONNX or single-op descriptions into EIC7700 `.model` files. The next
practical step for real NPU validation is to provide either:

- a precompiled single-op `.model` plus a runtime wrapper, or
- the official compiler/toolchain path used to generate `.model` files.

Until then, the framework should be used in `fallback` or `compare` mode for
API/golden-path development, and `npu` mode should be enabled only when a real
`VLLM_P550_NPU_RUNTIME_CLASS` is available.
