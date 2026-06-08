# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a matrix of V1 dummy GPU latency benchmarks.

This helper wraps `vllm bench latency --dummy-gpu-execution` for several
input/output length pairs and writes JSON, CSV, and Markdown summaries.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-lens", default="32,128,512,1024")
    parser.add_argument("--output-lens", default="64,128,512")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-iters-warmup", type=int, default=1)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--dummy-prefill-delay-ms", type=float, default=5.0)
    parser.add_argument("--dummy-decode-delay-ms", type=float, default=1.0)
    parser.add_argument("--dummy-token-id", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def run_case(
    args: argparse.Namespace,
    input_len: int,
    output_len: int,
    output_dir: Path,
) -> dict[str, Any]:
    case_json = output_dir / f"dummy_gpu_i{input_len}_o{output_len}.json"
    command = [
        "vllm",
        "bench",
        "latency",
        "--model",
        args.model,
        "--load-format",
        "dummy",
        "--enforce-eager",
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--batch-size",
        str(args.batch_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--num-iters-warmup",
        str(args.num_iters_warmup),
        "--num-iters",
        str(args.num_iters),
        "--dummy-gpu-execution",
        "--dummy-prefill-delay-ms",
        str(args.dummy_prefill_delay_ms),
        "--dummy-decode-delay-ms",
        str(args.dummy_decode_delay_ms),
        "--dummy-token-id",
        str(args.dummy_token_id),
        "--output-json",
        str(case_json),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed for input_len={input_len}, "
            f"output_len={output_len}\nSTDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    with open(case_json) as f:
        result = json.load(f)
    dummy = result["dummy_gpu_latency"]
    return {
        "input_len": input_len,
        "output_len": output_len,
        "avg_latency_s": result["avg_latency"],
        "ttft_avg_ms": dummy["ttft_ms"]["avg"],
        "ttft_p90_ms": dummy["ttft_ms"]["p90"],
        "tpot_ms": dummy["tpot_ms"],
        "decode_tps": dummy["decoding_tps"],
        "decode_step_avg_ms": dummy["decode_latency_ms"]["avg"],
        "decode_step_p90_ms": dummy["decode_latency_ms"]["p90"],
        "decode_token_count": dummy["decode_token_count"],
        "json": str(case_json),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "input_len",
        "output_len",
        "avg_latency_s",
        "ttft_avg_ms",
        "ttft_p90_ms",
        "tpot_ms",
        "decode_tps",
        "decode_step_avg_ms",
        "decode_step_p90_ms",
    ]
    with open(path, "w") as f:
        f.write("# Dummy GPU Latency Matrix\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            values = []
            for header in headers:
                value = row[header]
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")


def main() -> None:
    args = parse_args()
    input_lens = parse_int_list(args.input_lens)
    output_lens = parse_int_list(args.output_lens)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        run_case(args, input_len, output_len, output_dir)
        for input_len in input_lens
        for output_len in output_lens
    ]
    with open(output_dir / "dummy_gpu_latency_matrix.json", "w") as f:
        json.dump(rows, f, indent=4)
    write_csv(rows, output_dir / "dummy_gpu_latency_matrix.csv")
    write_markdown(rows, output_dir / "dummy_gpu_latency_matrix.md")


if __name__ == "__main__":
    sys.exit(main())
