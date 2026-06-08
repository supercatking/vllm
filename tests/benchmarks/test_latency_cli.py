# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
from types import SimpleNamespace

import pytest

from vllm.benchmarks.latency import (
    _summarize_dummy_gpu_records,
    _validate_dummy_outputs,
)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.benchmark
def test_bench_latency():
    command = [
        "vllm",
        "bench",
        "latency",
        "--model",
        MODEL_NAME,
        "--input-len",
        "32",
        "--output-len",
        "1",
        "--enforce-eager",
        "--load-format",
        "dummy",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


def test_dummy_gpu_latency_summary():
    args = SimpleNamespace(
        dummy_prefill_delay_ms=5.0,
        dummy_decode_delay_ms=1.0,
        dummy_token_id=0,
    )
    records = [
        {
            "step_type": "prefill",
            "latency_ms": 5.5,
            "num_reqs": 2,
        },
        {
            "step_type": "decode",
            "latency_ms": 1.2,
            "num_reqs": 2,
        },
        {
            "step_type": "decode",
            "latency_ms": 1.0,
            "num_reqs": 2,
        },
    ]

    summary = _summarize_dummy_gpu_records(args, records)

    assert summary["ttft_ms"]["count"] == 1
    assert summary["decode_latency_ms"]["count"] == 2
    assert summary["decode_token_count"] == 4
    assert summary["tpot_ms"] == pytest.approx(0.55)
    assert summary["decoding_tps"] == pytest.approx(1818.1818, rel=1e-4)


def test_validate_dummy_outputs():
    args = SimpleNamespace(
        dummy_gpu_execution=True,
        use_beam_search=False,
        output_len=3,
        dummy_token_id=0,
    )
    outputs = [
        SimpleNamespace(outputs=[SimpleNamespace(token_ids=[0, 0, 0])]),
        SimpleNamespace(outputs=[SimpleNamespace(token_ids=[0, 0, 0])]),
    ]

    _validate_dummy_outputs(args, outputs)

    bad_outputs = [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[0, 1, 0])])]
    with pytest.raises(AssertionError):
        _validate_dummy_outputs(args, bad_outputs)
