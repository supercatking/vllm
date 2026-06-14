# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.p550.npu.config import P550NpuConfig
from vllm.p550.npu.golden import (
    TensorOutput,
    compare_outputs,
    comparisons_ok,
    serialize_comparisons,
    write_jsonl_report,
)
from vllm.p550.npu.runtime import P550NpuRuntime, load_runtime


@dataclass(frozen=True)
class P550NpuOp:
    name: str
    golden_fn: Callable[..., TensorOutput]
    aliases: tuple[str, ...] = ()
    description: str = ""


_OPS: dict[str, P550NpuOp] = {}
_ALIASES: dict[str, str] = {}


def register_op(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    description: str = "",
) -> Callable[[Callable[..., TensorOutput]], Callable[..., TensorOutput]]:
    def decorator(fn: Callable[..., TensorOutput]) -> Callable[..., TensorOutput]:
        op = P550NpuOp(
            name=name,
            golden_fn=fn,
            aliases=aliases,
            description=description,
        )
        _OPS[name] = op
        for alias in aliases:
            _ALIASES[alias] = name
        return fn

    return decorator


def list_ops() -> list[P550NpuOp]:
    return [_OPS[name] for name in sorted(_OPS)]


def get_op(name: str) -> P550NpuOp:
    canonical_name = _ALIASES.get(name, name)
    try:
        return _OPS[canonical_name]
    except KeyError as exc:
        available = ", ".join(sorted(_OPS))
        raise KeyError(
            f"Unknown P550 NPU op {name!r}. Available ops: {available}"
        ) from exc


def _tensor_summaries(
    output: TensorOutput | list[torch.Tensor],
) -> list[dict[str, Any]]:
    from vllm.p550.npu.golden import flatten_outputs

    return [
        {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        for tensor in flatten_outputs(output)
    ]


def _raise_compare_error(record: dict[str, Any]) -> None:
    raise AssertionError(
        "P550 NPU golden comparison failed: "
        f"op={record['op']} comparisons={record.get('comparisons')}"
    )


def run_op(
    name: str,
    *inputs: torch.Tensor,
    config: P550NpuConfig | None = None,
    runtime: P550NpuRuntime | None = None,
    collect_report: bool = False,
    **attrs: Any,
) -> TensorOutput | tuple[TensorOutput, dict[str, Any]]:
    cfg = config or P550NpuConfig.from_env()
    op = get_op(name)
    if not cfg.allows_op(op.name):
        raise RuntimeError(
            f"P550 NPU op {op.name!r} is disabled by VLLM_P550_NPU_OPS."
        )

    active_runtime = runtime or load_runtime(cfg)
    record: dict[str, Any] = {
        "op": op.name,
        "backend": cfg.backend,
        "runtime": getattr(active_runtime, "name", active_runtime.__class__.__name__),
        "attrs": attrs,
        "inputs": _tensor_summaries(list(inputs)),
    }

    if cfg.backend == "off":
        start = time.perf_counter()
        actual = op.golden_fn(*inputs, **attrs)
        record["cpu_golden_latency_ms"] = (time.perf_counter() - start) * 1000.0
        record["outputs"] = _tensor_summaries(actual)
        record["ok"] = True
    elif cfg.backend == "fallback":
        start = time.perf_counter()
        actual = active_runtime.run_op(op.name, inputs, attrs)
        record["runtime_latency_ms"] = (time.perf_counter() - start) * 1000.0
        record["outputs"] = _tensor_summaries(actual)
        record["ok"] = True
    elif cfg.backend == "compare":
        runtime_start = time.perf_counter()
        actual = active_runtime.run_op(op.name, inputs, attrs)
        record["runtime_latency_ms"] = (time.perf_counter() - runtime_start) * 1000.0

        golden_start = time.perf_counter()
        expected = op.golden_fn(*inputs, **attrs)
        record["cpu_golden_latency_ms"] = (
            time.perf_counter() - golden_start
        ) * 1000.0

        comparisons = compare_outputs(actual, expected, atol=cfg.atol, rtol=cfg.rtol)
        record["comparisons"] = serialize_comparisons(comparisons)
        record["outputs"] = _tensor_summaries(actual)
        record["ok"] = comparisons_ok(comparisons)
        if not record["ok"] and cfg.strict_compare:
            _raise_compare_error(record)
    elif cfg.backend == "npu":
        start = time.perf_counter()
        actual = active_runtime.run_op(op.name, inputs, attrs)
        record["runtime_latency_ms"] = (time.perf_counter() - start) * 1000.0
        record["outputs"] = _tensor_summaries(actual)
        record["ok"] = True
    else:
        raise AssertionError(f"Unhandled P550 NPU backend: {cfg.backend}")

    if cfg.report_path:
        write_jsonl_report(cfg.report_path, record)
    if collect_report:
        return actual, record
    return actual
