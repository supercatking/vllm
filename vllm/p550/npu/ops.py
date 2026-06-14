# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.p550.npu.registry import register_op

_REGISTERED = False


def register_builtin_ops() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    @register_op("add", description="Elementwise add with broadcasting.")
    def _add(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.add(lhs, rhs)

    @register_op("matmul", aliases=("linear",), description="Matrix multiply.")
    def _matmul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.matmul(lhs, rhs)

    @register_op(
        "silu_mul",
        description="SwiGLU-style fused silu(x) * gate helper.",
    )
    def _silu_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return F.silu(x) * gate

    @register_op(
        "rms_norm",
        description="RMSNorm forward without residual add.",
    )
    def _rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        x_float = x.float()
        weight_float = weight.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        output = x_float * torch.rsqrt(variance + eps)
        output = output * weight_float
        return output.to(dtype=x.dtype)

    _REGISTERED = True
