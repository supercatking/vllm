# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.platforms.cpu import CpuPlatform

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


class P550NpuPlatform(CpuPlatform):
    """P550 NPU development platform.

    The first bring-up stage deliberately keeps ``device_type == "cpu"``.
    Full-model inference still uses the validated P550 CPU backend, while
    individual operators can be redirected through ``vllm.p550.npu`` for
    NPU-vs-CPU golden comparison.
    """

    device_name: str = "p550-npu-dev"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    dist_backend: str = "gloo"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return cls.device_name

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        super().check_and_update_config(vllm_config)

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls in {
            "auto",
            "vllm.v1.worker.cpu_worker.CPUWorker",
        }:
            parallel_config.worker_cls = (
                "vllm.v1.worker.p550_npu_worker.P550NpuWorker"
            )

        logger.info(
            "P550 NPU development platform is active. Full-model execution "
            "uses CPU fallback; single-op NPU experiments are controlled by "
            "VLLM_P550_NPU_BACKEND."
        )
