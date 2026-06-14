# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from typing import Any, Protocol

import torch

from vllm.p550.npu.config import P550NpuConfig

TensorArgs = tuple[torch.Tensor, ...]


class P550NpuRuntime(Protocol):
    name: str

    def run_op(
        self,
        op_name: str,
        inputs: TensorArgs,
        attrs: dict[str, Any],
    ) -> Any:
        ...


class CpuFallbackRuntime:
    name = "cpu_fallback"

    def run_op(
        self,
        op_name: str,
        inputs: TensorArgs,
        attrs: dict[str, Any],
    ) -> Any:
        from vllm.p550.npu.registry import get_op

        op = get_op(op_name)
        return op.golden_fn(*inputs, **attrs)


class UnavailableNpuRuntime:
    name = "unavailable_npu"

    def run_op(
        self,
        op_name: str,
        inputs: TensorArgs,
        attrs: dict[str, Any],
    ) -> Any:
        raise RuntimeError(
            "VLLM_P550_NPU_BACKEND=npu requires "
            "VLLM_P550_NPU_RUNTIME_CLASS=<module.Class>."
        )


def _load_class(qualname: str):
    module_name, _, class_name = qualname.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(
            "VLLM_P550_NPU_RUNTIME_CLASS must be a fully qualified class name."
        )
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_runtime(config: P550NpuConfig) -> P550NpuRuntime:
    if config.runtime_class:
        runtime_cls = _load_class(config.runtime_class)
        return runtime_cls()
    if config.backend in {"off", "fallback", "compare"}:
        return CpuFallbackRuntime()
    return UnavailableNpuRuntime()
