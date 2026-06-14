# P550 NPU Single-Op Backend Development Framework

This document describes the first-stage P550 NPU development path in this
branch. The goal is to let compiler developers implement and validate
individual NPU operators while preserving the existing P550 CPU backend as the
golden fallback for full vLLM inference.

## Goals

- Keep the current P550 CPU vLLM backend usable for end-to-end inference.
- Add an explicit P550 NPU development platform that is enabled only by
  environment variables.
- Provide a small, testable single-op runtime interface for NPU bring-up.
- Compare NPU outputs against CPU golden functions with JSONL reporting.
- Allow several compiler developers to work on independent operators without
  changing scheduler, model loading, or CPU fallback code.

## Non-Goals

- This is not a complete full-model NPU backend.
- This does not replace the existing P550 CPU backend.
- This does not hide correctness failures. In compare mode, every enabled NPU
  operator should be checked against CPU golden output.

## Activation Model

The development platform is opt-in:

```bash
export VLLM_P550_NPU_BACKEND=compare
```

Supported backend modes:

- `off`: do not use the NPU framework; run CPU golden functions directly.
- `fallback`: route through the runtime adapter, using CPU fallback if no real
  runtime is provided.
- `compare`: run the configured runtime and CPU golden function, then compare
  outputs.
- `npu`: run only the configured NPU runtime. This requires
  `VLLM_P550_NPU_RUNTIME_CLASS`.

For full vLLM serving on P550, the platform still reports `device_type="cpu"`.
This is deliberate: the known-good CPU backend remains the full-model fallback,
and NPU work starts at single-op granularity.

## Environment Variables

```bash
export VLLM_P550_NPU_BACKEND=compare
export VLLM_P550_NPU_OPS=matmul,rms_norm
export VLLM_P550_NPU_ATOL=1e-4
export VLLM_P550_NPU_RTOL=1e-4
export VLLM_P550_NPU_STRICT=1
export VLLM_P550_NPU_REPORT_PATH=/tmp/p550_npu_ops.jsonl
export VLLM_P550_NPU_RUNTIME_CLASS=my_runtime.P550Runtime
```

`VLLM_P550_NPU_RUNTIME_CLASS` must name a class with this method:

```python
def run_op(self, op_name: str, inputs: tuple[torch.Tensor, ...], attrs: dict):
    ...
```

The method should return a tensor or a tuple/list of tensors. In `compare`
mode, the framework compares the returned tensors with CPU golden outputs.

## Built-In Operators

The initial registry includes:

- `add`
- `matmul` (`linear` alias)
- `silu_mul`
- `rms_norm`

Each operator is registered in `vllm/p550/npu/ops.py`. Compiler developers
should add one operator at a time with a CPU golden function and tests.

## CLI Smoke Tests

List registered operators:

```bash
python -m vllm.p550.npu.cli list
```

Run a matmul comparison through CPU fallback:

```bash
python -m vllm.p550.npu.cli run \
  --op matmul \
  --shape 2x4 \
  --rhs-shape 4x3 \
  --dtype float32 \
  --backend compare \
  --report /tmp/p550_npu_ops.jsonl
```

The command prints a JSON summary and appends a JSONL record containing:

- operator name
- backend mode
- runtime name
- input and output tensor metadata
- runtime latency
- CPU golden latency
- per-output max absolute and relative error
- pass/fail status

## Development Workflow

1. Add or update a CPU golden implementation in `vllm/p550/npu/ops.py`.
2. Add a real runtime implementation outside the registry.
3. Set `VLLM_P550_NPU_RUNTIME_CLASS` to the runtime class.
4. Run the CLI with `--backend compare --strict`.
5. Inspect JSONL results and fix compiler/runtime mismatches.
6. Only after single-op correctness is stable should the op be considered for
   integration into a real model execution path.

## EIC7700 Status From Peter Probe

The `peter` board exposes `/dev/npu0`, ESWIN NPU/DSP kernel modules,
`/usr/include/essdk/es_npu_interface.h`, `libedla_runtime.so`, and
`/opt/eswin/bin/es_run_model`. This is sufficient for runtime validation with
precompiled `.model` files.

The current board image did not expose the offline compiler toolchain needed to
turn arbitrary ONNX/single-op descriptions into EIC7700 `.model` files. Until
that compiler is installed, real NPU testing should target precompiled models
or a runtime wrapper supplied by the compiler team.

## Test Commands

```bash
python -m pytest tests/p550/test_npu_framework.py -q
python -m vllm.p550.npu.cli list
python -m vllm.p550.npu.cli run --op add --shape 4x4 --backend compare
```

## Separation From CPU Backend

The existing P550 CPU backend remains the golden reference and full-model
fallback. The NPU framework is isolated under `vllm.p550.npu`, and the worker
shim is only selected when the P550 NPU platform is explicitly requested.
This prevents dummy or experimental NPU code from being confused with the
already validated CPU inference path.
