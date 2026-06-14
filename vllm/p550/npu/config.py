# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass, replace
from typing import Literal

BackendMode = Literal["off", "fallback", "compare", "npu"]


def _parse_backend(value: str | None) -> BackendMode:
    normalized = (value or "off").strip().lower().replace("-", "_")
    if normalized in {"", "0", "false", "no", "off", "disabled"}:
        return "off"
    if normalized in {"1", "true", "yes", "on"}:
        return "compare"
    if normalized in {"cpu", "fallback", "cpu_fallback"}:
        return "fallback"
    if normalized in {"golden", "compare", "golden_compare"}:
        return "compare"
    if normalized in {"npu", "p550_npu"}:
        return "npu"
    raise ValueError(
        "Unsupported VLLM_P550_NPU_BACKEND value "
        f"{value!r}; expected off, fallback, compare, or npu."
    )


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_ops(value: str | None) -> frozenset[str] | None:
    if value is None or value.strip() == "":
        return None
    return frozenset(op.strip() for op in value.split(",") if op.strip())


@dataclass(frozen=True)
class P550NpuConfig:
    """Runtime controls for P550 NPU single-op bring-up."""

    backend: BackendMode = "off"
    enabled_ops: frozenset[str] | None = None
    atol: float = 1e-4
    rtol: float = 1e-4
    strict_compare: bool = False
    report_path: str | None = None
    runtime_class: str | None = None

    @classmethod
    def from_env(cls) -> "P550NpuConfig":
        return cls(
            backend=_parse_backend(os.getenv("VLLM_P550_NPU_BACKEND")),
            enabled_ops=_parse_ops(os.getenv("VLLM_P550_NPU_OPS")),
            atol=float(os.getenv("VLLM_P550_NPU_ATOL", "1e-4")),
            rtol=float(os.getenv("VLLM_P550_NPU_RTOL", "1e-4")),
            strict_compare=_parse_bool(os.getenv("VLLM_P550_NPU_STRICT")),
            report_path=os.getenv("VLLM_P550_NPU_REPORT_PATH") or None,
            runtime_class=os.getenv("VLLM_P550_NPU_RUNTIME_CLASS") or None,
        )

    def allows_op(self, op_name: str) -> bool:
        return self.enabled_ops is None or op_name in self.enabled_ops

    def with_overrides(self, **kwargs) -> "P550NpuConfig":
        return replace(self, **kwargs)
