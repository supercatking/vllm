# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic host scheduler benchmark.

This benchmark runs the vLLM V1 scheduler and model-runner path while replacing
real accelerator execution with deterministic dummy delays. It measures
CPU-visible host/scheduler/control-plane overhead, not model quality, CPU model
execution, or real GPU/NPU kernel performance.
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import geometric_mean
from typing import Any

import numpy as np


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-profile", default=None)
    parser.add_argument("--synthetic-backend", choices=["sleep"], default="sleep")
    parser.add_argument("--input-lens", default="32,128,512,1024")
    parser.add_argument("--output-lens", default="64,128,512")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dummy-prefill-ms", type=float, default=5.0)
    parser.add_argument("--dummy-decode-ms", type=float, default=1.0)
    parser.add_argument("--dummy-output-token-id", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--reference-json", default=None)


def _parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return values


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.array(values, dtype=np.float64)
    p50, p90, p99 = np.percentile(arr, [50, 90, 99])
    return {
        "count": int(arr.size),
        "avg": float(np.mean(arr)),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
    }


def _run_text(command: list[str], timeout: int = 20) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip()
        return output if output else ""
    except Exception as exc:
        return f"unavailable: {exc}"


def _env_report() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.replace("\n", " "),
        "hostname": platform.node(),
        "vllm_commit": _run_text(["git", "rev-parse", "HEAD"]),
        "vllm_branch": _run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "kernel": _run_text(["uname", "-a"]),
        "cpu": _run_text(["bash", "-lc", "lscpu | sed -n '1,35p'"]),
        "nvidia_smi": _run_text(["bash", "-lc", "nvidia-smi 2>&1 | sed -n '1,40p'"]),
        "taskset": _run_text(["bash", "-lc", "taskset -pc $$ 2>&1"]),
    }


def _case_command(
    args: argparse.Namespace,
    input_len: int,
    output_len: int,
    case_json: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
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
        str(args.warmup),
        "--num-iters",
        str(args.repeats),
        "--dummy-gpu-execution",
        "--dummy-prefill-delay-ms",
        str(args.dummy_prefill_ms),
        "--dummy-decode-delay-ms",
        str(args.dummy_decode_ms),
        "--dummy-token-id",
        str(args.dummy_output_token_id),
        "--output-json",
        str(case_json),
    ]
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    return command


def _summarize_case(
    args: argparse.Namespace,
    input_len: int,
    output_len: int,
    case_json: Path,
    completed: subprocess.CompletedProcess[str],
    elapsed_s: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": f"i{input_len}_o{output_len}",
        "input_tokens_target": input_len,
        "output_tokens_target": output_len,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "synthetic_backend": args.synthetic_backend,
        "dummy_prefill_ms": args.dummy_prefill_ms,
        "dummy_decode_ms": args.dummy_decode_ms,
        "dummy_output_token_id": args.dummy_output_token_id,
        "success": False,
        "invalid_reason": "",
        "raw_json": str(case_json),
        "wall_time_s": elapsed_s,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        row["invalid_reason"] = f"latency command exited {completed.returncode}"
        return row
    try:
        result = json.loads(case_json.read_text())
        dummy = result["dummy_gpu_latency"]
        prefill = dummy["prefill_latency_ms"]
        decode = dummy["decode_latency_ms"]
        ttft = dummy["ttft_ms"]
        avg_e2e_ms = float(result["avg_latency"]) * 1000.0
        decode_token_count = int(dummy["decode_token_count"])
        prefill_actual_ms = float(prefill["avg"])
        decode_step_actual_ms = float(decode["avg"])
        decode_steps_per_request = max(0, output_len - 1)
        dummy_device_ms = prefill_actual_ms + (
            decode_step_actual_ms * decode_steps_per_request
        )
        host_adjusted_e2e_ms = max(0.0, avg_e2e_ms - dummy_device_ms)
        host_adjusted_ttft_ms = max(0.0, float(ttft["avg"]) - prefill_actual_ms)
        row.update(
            {
                "success": True,
                "avg_e2e_ms": avg_e2e_ms,
                "ttft_avg_ms": float(ttft["avg"]),
                "ttft_p50_ms": float(ttft["p50"]),
                "ttft_p90_ms": float(ttft["p90"]),
                "ttft_p99_ms": float(ttft["p99"]),
                "prefill_avg_ms": float(prefill["avg"]),
                "prefill_p50_ms": float(prefill["p50"]),
                "prefill_p90_ms": float(prefill["p90"]),
                "prefill_p99_ms": float(prefill["p99"]),
                "decode_avg_ms": float(decode["avg"]),
                "decode_p50_ms": float(decode["p50"]),
                "decode_p90_ms": float(decode["p90"]),
                "decode_p99_ms": float(decode["p99"]),
                "tpot_ms": float(dummy["tpot_ms"]),
                "decode_tps": float(dummy["decoding_tps"]),
                "decode_token_count": decode_token_count,
                "host_adjusted_e2e_ms": host_adjusted_e2e_ms,
                "host_adjusted_ttft_ms": host_adjusted_ttft_ms,
                "dummy_device_ms": dummy_device_ms,
                "latencies_s": result["latencies"],
            }
        )
    except Exception as exc:
        row["invalid_reason"] = f"failed to parse latency JSON: {exc}"
    return row


def _run_case(
    args: argparse.Namespace,
    input_len: int,
    output_len: int,
    output_dir: Path,
) -> dict[str, Any]:
    case_json = output_dir / "raw_logs" / f"synthetic_i{input_len}_o{output_len}.json"
    command = _case_command(args, input_len, output_len, case_json)
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    log_path = output_dir / "raw_logs" / f"synthetic_i{input_len}_o{output_len}.log"
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT:\n" + completed.stdout
        + "\nSTDERR:\n" + completed.stderr
    )
    return _summarize_case(args, input_len, output_len, case_json, completed, elapsed)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys and key not in {"stdout_tail", "stderr_tail", "latencies_s"}:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in keys} for row in rows])


def _score(rows: list[dict[str, Any]], reference_json: str | None) -> dict[str, Any]:
    if reference_json is None:
        return {"available": False, "reason": "no reference-json provided"}
    try:
        reference = json.loads(Path(reference_json).read_text())
        ref_cases = {case["case_id"]: case for case in reference["cases"]}
        ratios = []
        case_scores = {}
        for row in rows:
            if not row.get("success") or row["case_id"] not in ref_cases:
                continue
            ref_value = float(ref_cases[row["case_id"]]["host_adjusted_e2e_ms"])
            sut_value = float(row["host_adjusted_e2e_ms"])
            if ref_value > 0 and sut_value > 0:
                ratio = ref_value / sut_value
                ratios.append(ratio)
                case_scores[row["case_id"]] = ratio
        return {
            "available": bool(ratios),
            "serving_score_geomean": geometric_mean(ratios) if ratios else 0.0,
            "case_scores": case_scores,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _write_md(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    env: dict[str, Any],
    score: dict[str, Any],
    path: Path,
) -> None:
    headers = [
        "case_id",
        "input_tokens_target",
        "output_tokens_target",
        "success",
        "avg_e2e_ms",
        "ttft_avg_ms",
        "tpot_ms",
        "decode_tps",
        "host_adjusted_e2e_ms",
        "invalid_reason",
    ]
    lines = [
        "# vLLM Synthetic Host Scheduler Benchmark",
        "",
        "This report measures CPU-visible vLLM host/scheduler overhead with synthetic GPU execution. It does not measure model quality, CPU model execution, or real accelerator kernel performance.",
        "",
        "## Configuration",
        "",
        f"- model: `{args.model}`",
        f"- backend: `{args.synthetic_backend}`",
        f"- prefill delay: `{args.dummy_prefill_ms} ms`",
        f"- decode delay: `{args.dummy_decode_ms} ms`",
        f"- dummy output token: `{args.dummy_output_token_id}`",
        f"- batch size: `{args.batch_size}`",
        f"- repeats/warmup: `{args.repeats}/{args.warmup}`",
        "",
        "## Environment",
        "",
        f"- hostname: `{env.get('hostname')}`",
        f"- machine: `{env.get('machine')}`",
        f"- kernel: `{env.get('kernel')}`",
        f"- vLLM commit: `{env.get('vllm_commit')}`",
        "",
        "## Results",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value).replace("\n", " ")[:160])
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Score",
            "",
            "Reference-based scoring is optional.",
            "",
            "```json",
            json.dumps(score, indent=2, sort_keys=True),
            "```",
            "",
            "## Interpretation",
            "",
            "- TTFT tracks the CPU-visible completion of synthetic prefill.",
            "- TPOT and decode TPS track CPU-visible synthetic decode token return.",
            "- host_adjusted_e2e_ms subtracts observed synthetic device delay from raw end-to-end latency to approximate host overhead.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main(args: argparse.Namespace) -> None:
    if args.concurrency != 1:
        raise ValueError("v0.1 synthetic benchmark supports concurrency=1 only.")
    input_lens = _parse_int_list(args.input_lens)
    output_lens = _parse_int_list(args.output_lens)
    output_dir = Path(args.output_dir)
    (output_dir / "raw_logs").mkdir(parents=True, exist_ok=True)

    env = _env_report()
    run_config = {
        key: value
        for key, value in vars(args).items()
        if not callable(value) and not key.startswith("_")
    }
    (output_dir / "env_report.json").write_text(json.dumps(env, indent=2))
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    rows = [
        _run_case(args, input_len, output_len, output_dir)
        for input_len in input_lens
        for output_len in output_lens
    ]
    score = _score(rows, args.reference_json)
    result = {
        "benchmark": "vllm_synthetic_host_scheduler",
        "version": 1,
        "generated_at_unix": time.time(),
        "environment": env,
        "config": run_config,
        "score": score,
        "cases": rows,
        "summary": {
            "success_count": sum(1 for row in rows if row.get("success")),
            "failure_count": sum(1 for row in rows if not row.get("success")),
            "host_adjusted_e2e_ms": _stats(
                [float(row["host_adjusted_e2e_ms"]) for row in rows if row.get("success")]
            ),
            "tpot_ms": _stats([float(row["tpot_ms"]) for row in rows if row.get("success")]),
        },
    }

    json_path = Path(args.output_json) if args.output_json else output_dir / "matrix_results.json"
    csv_path = Path(args.output_csv) if args.output_csv else output_dir / "matrix_results.csv"
    md_path = Path(args.output_md) if args.output_md else output_dir / "analysis.md"
    json_path.write_text(json.dumps(result, indent=2))
    _write_csv(rows, csv_path)
    _write_md(args, rows, env, score, md_path)
    print(f"Wrote synthetic benchmark JSON: {json_path}")
    print(f"Wrote synthetic benchmark CSV: {csv_path}")
    print(f"Wrote synthetic benchmark report: {md_path}")
    if any(not row.get("success") for row in rows):
        raise SystemExit(1)
