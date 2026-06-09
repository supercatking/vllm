# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
from types import SimpleNamespace

import pytest

from vllm.benchmarks.synthetic import _parse_int_list, _summarize_case


def test_parse_int_list():
    assert _parse_int_list("32, 128,512") == [32, 128, 512]
    with pytest.raises(ValueError):
        _parse_int_list("")


def test_summarize_case_host_adjusted_metrics(tmp_path):
    case_json = tmp_path / "case.json"
    case_json.write_text(
        json.dumps(
            {
                "avg_latency": 0.020,
                "latencies": [0.020],
                "dummy_gpu_latency": {
                    "ttft_ms": {
                        "count": 1,
                        "avg": 5.5,
                        "p50": 5.5,
                        "p90": 5.5,
                        "p99": 5.5,
                    },
                    "prefill_latency_ms": {
                        "count": 1,
                        "avg": 5.0,
                        "p50": 5.0,
                        "p90": 5.0,
                        "p99": 5.0,
                    },
                    "decode_latency_ms": {
                        "count": 2,
                        "avg": 1.0,
                        "p50": 1.0,
                        "p90": 1.0,
                        "p99": 1.0,
                    },
                    "tpot_ms": 1.0,
                    "decoding_tps": 1000.0,
                    "decode_token_count": 2,
                },
            }
        )
    )
    args = SimpleNamespace(
        batch_size=1,
        concurrency=1,
        synthetic_backend="sleep",
        dummy_prefill_ms=5.0,
        dummy_decode_ms=1.0,
        dummy_output_token_id=0,
    )
    completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")

    row = _summarize_case(args, 32, 3, case_json, completed, 1.0)

    assert row["success"] is True
    assert row["avg_e2e_ms"] == pytest.approx(20.0)
    assert row["ttft_avg_ms"] == pytest.approx(5.5)
    assert row["tpot_ms"] == pytest.approx(1.0)
    assert row["decode_tps"] == pytest.approx(1000.0)
    assert row["host_adjusted_e2e_ms"] == pytest.approx(13.0)
    assert row["host_adjusted_ttft_ms"] == pytest.approx(0.5)
