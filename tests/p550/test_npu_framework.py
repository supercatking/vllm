# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from vllm.p550.npu.config import P550NpuConfig
from vllm.p550.npu.ops import register_builtin_ops
from vllm.p550.npu.registry import list_ops, run_op
from vllm.platforms import p550_npu_platform_plugin
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.p550_npu import P550NpuPlatform


class BadRuntime:
    name = "bad_runtime"

    def run_op(self, op_name, inputs, attrs):
        return torch.zeros_like(inputs[0])


def test_builtin_ops_are_registered():
    register_builtin_ops()
    assert {"add", "matmul", "rms_norm", "silu_mul"}.issubset(
        {op.name for op in list_ops()}
    )


def test_compare_backend_writes_jsonl_report(tmp_path: Path):
    report = tmp_path / "p550_npu.jsonl"
    lhs = torch.randn((2, 4), dtype=torch.float32)
    rhs = torch.randn((4, 3), dtype=torch.float32)
    config = P550NpuConfig(backend="compare", report_path=str(report))

    output, record = run_op(
        "matmul",
        lhs,
        rhs,
        config=config,
        collect_report=True,
    )

    assert record["ok"] is True
    assert output.shape == (2, 3)
    saved = json.loads(report.read_text(encoding="utf-8").splitlines()[0])
    assert saved["op"] == "matmul"
    assert saved["backend"] == "compare"
    assert saved["ok"] is True
    assert saved["comparisons"][0]["within_tolerance"] is True


def test_strict_compare_raises_on_bad_runtime():
    x = torch.ones((2, 2), dtype=torch.float32)
    y = torch.ones((2, 2), dtype=torch.float32)
    config = P550NpuConfig(backend="compare", strict_compare=True)

    with pytest.raises(AssertionError):
        run_op("add", x, y, config=config, runtime=BadRuntime())


def test_p550_platform_plugin_is_explicit_env_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_P550_NPU_BACKEND", raising=False)
    assert p550_npu_platform_plugin() is None

    monkeypatch.setenv("VLLM_P550_NPU_BACKEND", "compare")
    assert p550_npu_platform_plugin() == "vllm.platforms.p550_npu.P550NpuPlatform"


def test_p550_platform_uses_cpu_device_and_worker(monkeypatch):
    monkeypatch.setattr(
        CpuPlatform,
        "check_and_update_config",
        classmethod(lambda cls, config: None),
    )
    config = Mock()
    config.parallel_config.worker_cls = "auto"

    P550NpuPlatform.check_and_update_config(config)

    assert P550NpuPlatform.device_type == "cpu"
    assert config.parallel_config.worker_cls == (
        "vllm.v1.worker.p550_npu_worker.P550NpuWorker"
    )
