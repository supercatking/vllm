# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the latency of processing a single batch of requests."""

import argparse
import json
import os
import tempfile
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from vllm.benchmarks.lib.utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import BeamSearchParams

DUMMY_GPU_EXECUTION_ENV = "BENCH_DUMMY_GPU_EXECUTION"
DUMMY_GPU_PREFILL_DELAY_MS_ENV = "BENCH_DUMMY_GPU_PREFILL_DELAY_MS"
DUMMY_GPU_DECODE_DELAY_MS_ENV = "BENCH_DUMMY_GPU_DECODE_DELAY_MS"
DUMMY_GPU_TOKEN_ID_ENV = "BENCH_DUMMY_GPU_TOKEN_ID"
DUMMY_GPU_LATENCY_LOG_ENV = "BENCH_DUMMY_GPU_LATENCY_LOG"


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={"latency": results["latencies"]},
        extra_info={k: results[k] for k in ["avg_latency", "percentiles"]},
    )
    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def add_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument(
        "--num-iters", type=int, default=30, help="Number of iterations to run."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the latency results in JSON format.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=(
            "Do not detokenize responses (i.e. do not include "
            "detokenization time in the latency measurement)"
        ),
    )
    parser.add_argument(
        "--dummy-gpu-execution",
        action="store_true",
        help=(
            "Benchmark-only mode for the V1 GPU runner. The runner skips real "
            "model execution, sleeps for the configured prefill/decode delay, "
            "and returns the configured dummy token."
        ),
    )
    parser.add_argument(
        "--dummy-prefill-delay-ms",
        type=float,
        default=5.0,
        help="Dummy GPU prefill delay in milliseconds.",
    )
    parser.add_argument(
        "--dummy-decode-delay-ms",
        type=float,
        default=1.0,
        help="Dummy GPU decode delay in milliseconds.",
    )
    parser.add_argument(
        "--dummy-token-id",
        type=int,
        default=0,
        help="Token ID returned by dummy GPU execution.",
    )
    parser.add_argument(
        "--dummy-latency-output-json",
        type=str,
        default=None,
        help=(
            "Optional path to save dummy GPU prefill/decode latency summary "
            "as JSON. The same summary is also embedded in --output-json."
        ),
    )

    parser = EngineArgs.add_cli_args(parser)
    # V1 enables prefix caching by default which skews the latency
    # numbers. We need to disable prefix caching by default.
    parser.set_defaults(enable_prefix_caching=False)


def _set_dummy_gpu_env(args: argparse.Namespace) -> str | None:
    if not args.dummy_gpu_execution:
        return None

    dummy_latency_log = tempfile.NamedTemporaryFile(
        prefix="vllm_dummy_gpu_latency_", suffix=".jsonl", delete=False
    )
    dummy_latency_log.close()
    os.environ[DUMMY_GPU_EXECUTION_ENV] = "1"
    os.environ[DUMMY_GPU_PREFILL_DELAY_MS_ENV] = str(args.dummy_prefill_delay_ms)
    os.environ[DUMMY_GPU_DECODE_DELAY_MS_ENV] = str(args.dummy_decode_delay_ms)
    os.environ[DUMMY_GPU_TOKEN_ID_ENV] = str(args.dummy_token_id)
    os.environ[DUMMY_GPU_LATENCY_LOG_ENV] = dummy_latency_log.name
    return dummy_latency_log.name


def _clear_dummy_gpu_log(dummy_latency_log: str | None) -> None:
    if dummy_latency_log is not None:
        with open(dummy_latency_log, "w"):
            pass


def _load_dummy_gpu_records(dummy_latency_log: str | None) -> list[dict[str, Any]]:
    if dummy_latency_log is None or not os.path.exists(dummy_latency_log):
        return []

    records: list[dict[str, Any]] = []
    with open(dummy_latency_log) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _latency_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}

    latencies = np.array(values, dtype=np.float64)
    p50, p90, p99 = np.percentile(latencies, [50, 90, 99])
    return {
        "count": int(latencies.size),
        "avg": float(np.mean(latencies)),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
    }


def _summarize_dummy_gpu_records(
    args: argparse.Namespace, records: list[dict[str, Any]]
) -> dict[str, Any]:
    prefill = [
        float(record["latency_ms"])
        for record in records
        if record.get("step_type") in ("prefill", "prefill_chunk")
    ]
    prefill_chunk = [
        float(record["latency_ms"])
        for record in records
        if record.get("step_type") == "prefill_chunk"
    ]
    decode = [
        float(record["latency_ms"])
        for record in records
        if record.get("step_type") == "decode"
    ]
    total = [float(record["latency_ms"]) for record in records]
    decode_token_count = sum(
        int(record.get("num_reqs", 0))
        for record in records
        if record.get("step_type") == "decode"
    )
    decode_latency_s = sum(decode) / 1000.0
    decoding_tps = (
        float(decode_token_count / decode_latency_s) if decode_latency_s > 0 else 0.0
    )
    tpot_ms = (
        float((sum(decode) / decode_token_count)) if decode_token_count > 0 else 0.0
    )
    return {
        "dummy_gpu_execution": True,
        "dummy_prefill_delay_ms": args.dummy_prefill_delay_ms,
        "dummy_decode_delay_ms": args.dummy_decode_delay_ms,
        "dummy_token_id": args.dummy_token_id,
        "ttft_ms": _latency_stats(prefill),
        "tpot_ms": tpot_ms,
        "decode_token_count": decode_token_count,
        "decoding_tps": decoding_tps,
        "prefill_latency_ms": _latency_stats(prefill),
        "prefill_chunk_latency_ms": _latency_stats(prefill_chunk),
        "decode_latency_ms": _latency_stats(decode),
        "total_step_latency_ms": _latency_stats(total),
        "step_records": records,
    }


def _print_dummy_gpu_summary(summary: dict[str, Any]) -> None:
    print("Dummy GPU execution: enabled")
    print(
        "Dummy delays: "
        f"prefill={summary['dummy_prefill_delay_ms']} ms, "
        f"decode={summary['dummy_decode_delay_ms']} ms, "
        f"token={summary['dummy_token_id']}"
    )
    for key in (
        "ttft_ms",
        "prefill_latency_ms",
        "prefill_chunk_latency_ms",
        "decode_latency_ms",
        "total_step_latency_ms",
    ):
        stats = summary[key]
        print(
            f"{key}: count={stats['count']} avg={stats['avg']:.6f} "
            f"p50={stats['p50']:.6f} p90={stats['p90']:.6f} "
            f"p99={stats['p99']:.6f}"
        )
    print(f"TPOT: {summary['tpot_ms']:.6f} ms/token")
    print(f"Decoding TPS: {summary['decoding_tps']:.6f} tokens/s")


def _validate_dummy_outputs(args: argparse.Namespace, outputs: list[Any]) -> None:
    if not args.dummy_gpu_execution:
        return

    if args.use_beam_search:
        raise ValueError("Dummy GPU execution does not support beam search.")

    for req_idx, request_output in enumerate(outputs):
        completions = getattr(request_output, "outputs", [])
        for completion_idx, completion in enumerate(completions):
            token_ids = list(getattr(completion, "token_ids", []))
            if len(token_ids) != args.output_len:
                raise AssertionError(
                    "Dummy GPU execution returned an unexpected output length "
                    f"for request {req_idx}, completion {completion_idx}: "
                    f"expected {args.output_len}, got {len(token_ids)}."
                )
            bad_token_ids = [
                token_id for token_id in token_ids if token_id != args.dummy_token_id
            ]
            if bad_token_ids:
                raise AssertionError(
                    "Dummy GPU execution returned non-dummy token IDs "
                    f"for request {req_idx}, completion {completion_idx}: "
                    f"expected only {args.dummy_token_id}, got {token_ids}."
                )


def main(args: argparse.Namespace):
    dummy_latency_log = _set_dummy_gpu_env(args)
    if args.dummy_gpu_execution:
        args.async_scheduling = False
    engine_args = EngineArgs.from_cli_args(args)

    # Lazy import to avoid importing LLM when the bench command is not selected.
    from vllm import LLM, SamplingParams

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM.from_engine_args(engine_args)
    assert llm.llm_engine.model_config.max_model_len >= (
        args.input_len + args.output_len
    ), (
        "Please ensure that max_model_len is greater than"
        " the sum of input_len and output_len."
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )
    dummy_prompt_token_ids = np.random.randint(
        10000, size=(args.batch_size, args.input_len)
    )
    dummy_prompts: list[PromptType] = [
        {"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()
    ]

    def llm_generate():
        if not args.use_beam_search:
            outputs = llm.generate(
                dummy_prompts, sampling_params=sampling_params, use_tqdm=False
            )
            _validate_dummy_outputs(args, outputs)
            return outputs
        outputs = llm.beam_search(
            dummy_prompts,
            BeamSearchParams(
                beam_width=args.n,
                max_tokens=args.output_len,
                ignore_eos=True,
            )
        )
        _validate_dummy_outputs(args, outputs)
        return outputs

    def run_to_completion(do_profile: bool = False):
        if do_profile:
            llm.start_profile()
            llm_generate()
            llm.stop_profile()
        else:
            start_time = time.perf_counter()
            llm_generate()
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(do_profile=False)
    _clear_dummy_gpu_log(dummy_latency_log)

    if args.profile:
        profiler_config = engine_args.profiler_config
        if profiler_config.profiler == "torch":
            print(
                "Profiling with torch profiler (results will be saved to"
                f" {profiler_config.torch_profiler_dir})..."
            )
        elif profiler_config.profiler == "cuda":
            print("Profiling with cuda profiler ...")
        run_to_completion(do_profile=True)
        return

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Bench iterations"):
        latencies.append(run_to_completion(do_profile=False))
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f"Avg latency: {np.mean(latencies)} seconds")
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile} seconds")

    dummy_gpu_summary = None
    if args.dummy_gpu_execution:
        dummy_gpu_summary = _summarize_dummy_gpu_records(
            args, _load_dummy_gpu_records(dummy_latency_log)
        )
        _print_dummy_gpu_summary(dummy_gpu_summary)
        if args.dummy_latency_output_json:
            with open(args.dummy_latency_output_json, "w") as f:
                json.dump(dummy_gpu_summary, f, indent=4)

    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        if dummy_gpu_summary is not None:
            results["dummy_gpu_latency"] = dummy_gpu_summary
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)
