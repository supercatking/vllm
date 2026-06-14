# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

TensorOutput = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]


@dataclass
class TensorComparison:
    index: int
    shape: list[int]
    dtype: str
    expected_dtype: str
    max_abs_error: float
    max_rel_error: float
    within_tolerance: bool
    message: str = ""


def flatten_outputs(output: TensorOutput) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)):
        tensors: list[torch.Tensor] = []
        for item in output:
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"Unsupported output item type: {type(item)!r}")
            tensors.append(item)
        return tensors
    raise TypeError(f"Unsupported output type: {type(output)!r}")


def compare_outputs(
    actual: TensorOutput,
    expected: TensorOutput,
    *,
    atol: float,
    rtol: float,
) -> list[TensorComparison]:
    actual_tensors = flatten_outputs(actual)
    expected_tensors = flatten_outputs(expected)
    if len(actual_tensors) != len(expected_tensors):
        return [
            TensorComparison(
                index=-1,
                shape=[],
                dtype="",
                expected_dtype="",
                max_abs_error=float("inf"),
                max_rel_error=float("inf"),
                within_tolerance=False,
                message=(
                    "Output count mismatch: "
                    f"actual={len(actual_tensors)} expected={len(expected_tensors)}"
                ),
            )
        ]

    comparisons: list[TensorComparison] = []
    for idx, (actual_tensor, expected_tensor) in enumerate(
        zip(actual_tensors, expected_tensors)
    ):
        if actual_tensor.shape != expected_tensor.shape:
            comparisons.append(
                TensorComparison(
                    index=idx,
                    shape=list(actual_tensor.shape),
                    dtype=str(actual_tensor.dtype),
                    expected_dtype=str(expected_tensor.dtype),
                    max_abs_error=float("inf"),
                    max_rel_error=float("inf"),
                    within_tolerance=False,
                    message=(
                        "Shape mismatch: "
                        f"actual={list(actual_tensor.shape)} "
                        f"expected={list(expected_tensor.shape)}"
                    ),
                )
            )
            continue

        actual_float = actual_tensor.detach().to(dtype=torch.float64, device="cpu")
        expected_float = expected_tensor.detach().to(dtype=torch.float64, device="cpu")
        diff = (actual_float - expected_float).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        denom = expected_float.abs().clamp_min(1e-12)
        max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
        dtype_match = actual_tensor.dtype == expected_tensor.dtype
        value_match = torch.allclose(
            actual_float, expected_float, atol=atol, rtol=rtol
        )
        comparisons.append(
            TensorComparison(
                index=idx,
                shape=list(actual_tensor.shape),
                dtype=str(actual_tensor.dtype),
                expected_dtype=str(expected_tensor.dtype),
                max_abs_error=max_abs,
                max_rel_error=max_rel,
                within_tolerance=bool(dtype_match and value_match),
                message="" if dtype_match else "Dtype mismatch",
            )
        )
    return comparisons


def comparisons_ok(comparisons: list[TensorComparison]) -> bool:
    return all(item.within_tolerance for item in comparisons)


def serialize_comparisons(comparisons: list[TensorComparison]) -> list[dict[str, Any]]:
    return [asdict(item) for item in comparisons]


def write_jsonl_report(path: str, record: dict[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, sort_keys=True) + "\n")
