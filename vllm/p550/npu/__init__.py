# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""P550 NPU single-op development framework."""

from vllm.p550.npu.config import P550NpuConfig
from vllm.p550.npu.ops import register_builtin_ops
from vllm.p550.npu.registry import get_op, list_ops, register_op, run_op

register_builtin_ops()

__all__ = [
    "P550NpuConfig",
    "get_op",
    "list_ops",
    "register_builtin_ops",
    "register_op",
    "run_op",
]
