# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.p550.npu.config import P550NpuConfig
from vllm.p550.npu.ops import register_builtin_ops
from vllm.v1.worker.cpu_worker import CPUWorker

logger = init_logger(__name__)


class P550NpuWorker(CPUWorker):
    """CPU worker shim for P550 NPU single-op development.

    The worker preserves the known-good P550 CPU backend for end-to-end vLLM
    inference. Compiler developers can enable single-op NPU execution and
    CPU golden comparison via ``VLLM_P550_NPU_BACKEND`` without changing the
    scheduler or model-loading path.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        register_builtin_ops()
        self.p550_npu_config = P550NpuConfig.from_env()
        logger.info(
            "Initialized P550 NPU worker shim with backend=%s, enabled_ops=%s.",
            self.p550_npu_config.backend,
            sorted(self.p550_npu_config.enabled_ops or []),
        )
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
