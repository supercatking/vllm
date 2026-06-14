# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from typing import Any

import torch

from vllm.p550.npu.config import P550NpuConfig
from vllm.p550.npu.ops import register_builtin_ops
from vllm.p550.npu.registry import list_ops, run_op


def _parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(item) for item in value.lower().split("x") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid shape {value!r}; use forms like 1x128 or 128x128."
        ) from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(
            f"Invalid shape {value!r}; dimensions must be positive."
        )
    return shape


def _dtype(value: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return mapping[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"Unsupported dtype {value!r}; expected float32, bfloat16, or float16."
        ) from exc


def _make_inputs(
    args: argparse.Namespace,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    if args.op == "matmul":
        rhs_shape = args.rhs_shape or (args.shape[-1], args.shape[-1])
        lhs = torch.randn(args.shape, dtype=args.dtype, generator=generator)
        rhs = torch.randn(rhs_shape, dtype=args.dtype, generator=generator)
        return (lhs, rhs), {}

    if args.op in {"add", "silu_mul"}:
        lhs = torch.randn(args.shape, dtype=args.dtype, generator=generator)
        rhs = torch.randn(args.shape, dtype=args.dtype, generator=generator)
        return (lhs, rhs), {}

    if args.op == "rms_norm":
        x = torch.randn(args.shape, dtype=args.dtype, generator=generator)
        weight = torch.randn((args.shape[-1],), dtype=args.dtype, generator=generator)
        return (x, weight), {"eps": args.eps}

    raise ValueError(f"CLI input generation is not defined for op {args.op!r}.")


def _output_summary(output: Any) -> list[dict[str, Any]]:
    from vllm.p550.npu.golden import flatten_outputs

    return [
        {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "sum": float(tensor.float().sum().item()),
        }
        for tensor in flatten_outputs(output)
    ]


def cmd_list(_: argparse.Namespace) -> int:
    register_builtin_ops()
    for op in list_ops():
        aliases = f" aliases={','.join(op.aliases)}" if op.aliases else ""
        print(f"{op.name}{aliases}: {op.description}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    register_builtin_ops()
    inputs, attrs = _make_inputs(args)
    config = P550NpuConfig.from_env().with_overrides(
        backend=args.backend,
        report_path=args.report,
        atol=args.atol,
        rtol=args.rtol,
        strict_compare=args.strict,
    )
    output, record = run_op(
        args.op,
        *inputs,
        config=config,
        collect_report=True,
        **attrs,
    )
    print(
        json.dumps(
            {
                "op": args.op,
                "backend": config.backend,
                "ok": record["ok"],
                "record": record,
                "outputs": _output_summary(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if record["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    register_builtin_ops()
    parser = argparse.ArgumentParser(
        description="P550 NPU single-op golden comparison helper."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered ops.")
    list_parser.set_defaults(func=cmd_list)

    run_parser = subparsers.add_parser("run", help="Run one registered op.")
    run_parser.add_argument(
        "--op",
        required=True,
        choices=[op.name for op in list_ops()],
    )
    run_parser.add_argument("--shape", type=_parse_shape, default=(4, 4))
    run_parser.add_argument("--rhs-shape", type=_parse_shape, default=None)
    run_parser.add_argument("--dtype", type=_dtype, default=torch.float32)
    run_parser.add_argument(
        "--backend",
        choices=["off", "fallback", "compare", "npu"],
        default="compare",
    )
    run_parser.add_argument("--report", default=None)
    run_parser.add_argument("--atol", type=float, default=1e-4)
    run_parser.add_argument("--rtol", type=float, default=1e-4)
    run_parser.add_argument("--strict", action="store_true")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--eps", type=float, default=1e-6)
    run_parser.set_defaults(func=cmd_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
