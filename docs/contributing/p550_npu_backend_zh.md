# P550 vLLM NPU 单算子开发框架说明

文档日期：2026-06-15

目标分支：`supercatking/vllm:p550dev`

相关提交：`81a0dabfdc5cd0c27df8b5df9607a0cb8ecf7a14`

## 1. 背景

当前 `p550dev` 分支已经验证过一条 P550 上的 CPU backend vLLM 推理路径。这个路径的价值是：

- 能在 P550 RISC-V Linux 环境中跑通 vLLM。
- 能使用 CPU backend 完成小模型推理。
- 能作为后续 NPU backend 开发时的 full-model fallback 和 golden reference。

下一阶段的目标不是立即实现完整的 P550 NPU full-model backend，而是先提供一套 **NPU 单算子开发框架**：

- 编译器开发人员可以围绕单个算子开发 NPU kernel/runtime。
- 每个 NPU 算子都可以随时与 CPU golden 输出对比。
- vLLM 原有 CPU backend 不被破坏。
- 多个开发人员可以并行开发不同算子，减少互相踩代码的风险。
- 后续可以逐步从单算子验证推进到小 subgraph，再推进到模型执行路径集成。

这份文档整理了本轮已经实现和验证的工作，并说明编译器开发人员如何基于这个框架继续开发。

## 2. 总体结论

本轮已经在 `p550dev` 分支实现：

- 新增 `vllm.p550.npu` 单算子开发框架。
- 新增 P550 NPU dev platform shim。
- 新增 P550 NPU worker shim。
- 新增 CPU golden comparison、runtime adapter、JSONL report。
- 新增 CLI，支持本地和板端快速验证。
- 新增单元测试。
- 新增英文设计文档和验证记录。

本轮验证结果：

- 本机 WSL：单元测试 `5 passed`。
- 本机 WSL：`rms_norm` CLI compare 通过，误差为 `0.0`。
- 本机 WSL：`compileall` 通过。
- 本机 WSL：`git diff --check` 通过。
- peter：使用已有 `/home/ubuntu/vllm/.venv-p550/bin/python` 跑 `matmul/add` compare smoke，通过，误差为 `0.0`。
- dude：SSH 不通，错误为 `No route to host`，未执行验证。

对 EIC7700 NPU 的判断：

- peter 具备真实 NPU runtime 验证条件：设备、驱动、runtime library、头文件、sample runner 都存在。
- peter 暂未发现离线编译器 `eaac` 或等价模型转换工具，因此当前无法确认板上能完成 “ONNX/单算子源码 -> EIC7700 `.model`” 的完整编译闭环。
- 因此当前最合理路线是：先使用本框架做 API、shape、dtype、golden compare 和 runtime adapter 开发；真实 NPU 验证先围绕预编译 `.model` 或编译器团队提供的 runtime wrapper 展开。

## 3. 设计目标

### 3.1 保留 CPU backend 作为可信基线

现有 P550 CPU backend 是当前可运行、可验证的 full-model 路径。NPU 开发初期最重要的原则是：

- 不破坏 CPU backend。
- 不让实验性 NPU 路径默认接管模型推理。
- 所有 NPU 单算子输出都能与 CPU golden 对齐。

因此本轮实现中，P550 NPU dev platform 仍然保持：

```text
device_type = "cpu"
dispatch_key = "CPU"
dist_backend = "gloo"
```

这是一个有意设计，不是遗漏。这样可以让 vLLM 的模型加载、scheduler、worker 生命周期继续沿用 CPU backend，同时为 NPU 单算子开发提供独立入口。

### 3.2 NPU 框架必须显式启用

默认情况下，普通 CPU backend 不会启用 NPU dev platform。只有显式设置以下环境变量之一时才会启用：

```bash
export VLLM_P550_NPU_BACKEND=compare
```

或：

```bash
export VLLM_TARGET_DEVICE=p550_npu
```

这样可以避免把已经验证过的 CPU backend 与实验性 NPU/dummy/fallback 路径混淆。

### 3.3 从单算子开始，而不是直接 full model

完整 LLM 推理涉及：

- attention
- matmul/linear
- RMSNorm
- activation/fused activation
- KV cache
- sampling
- scheduler/control plane
- tokenizer/detokenizer
- memory management

如果一开始就把完整模型切到 NPU，问题会被混在一起：模型结构、compiler、runtime、memory layout、operator correctness、scheduler 都可能同时出错，定位成本极高。

本框架先把边界压缩到单算子：

```text
torch.Tensor inputs
        |
        v
P550 NPU runtime adapter
        |
        v
torch.Tensor outputs
        |
        v
CPU golden compare
```

这样每个算子都能独立验证 shape、dtype、数值误差和 runtime 行为。

### 3.4 支持多人并行开发

考虑到可能有 10 个编译器开发人员并行工作，框架按以下方式拆分：

- 算子注册：集中在 `vllm/p550/npu/ops.py`。
- runtime adapter：通过 `VLLM_P550_NPU_RUNTIME_CLASS` 注入，不要求所有人改同一个核心文件。
- golden compare：集中在 `vllm/p550/npu/golden.py`。
- 执行调度：集中在 `vllm/p550/npu/registry.py`。
- platform/worker shim：只做启用与隔离，不承载具体算子逻辑。

这样不同开发人员可以分别负责不同算子或 runtime wrapper，减少冲突。

## 4. 框架结构

### 4.1 代码目录

本轮新增或修改的主要文件：

```text
vllm/platforms/__init__.py
vllm/platforms/p550_npu.py
vllm/v1/worker/p550_npu_worker.py

vllm/p550/__init__.py
vllm/p550/npu/__init__.py
vllm/p550/npu/config.py
vllm/p550/npu/runtime.py
vllm/p550/npu/registry.py
vllm/p550/npu/golden.py
vllm/p550/npu/ops.py
vllm/p550/npu/cli.py

tests/p550/test_npu_framework.py

docs/contributing/p550_npu_backend.md
docs/contributing/p550_npu_framework_validation.md
docs/contributing/p550_npu_backend_zh.md
```

### 4.2 模块职责

| 文件 | 职责 |
| --- | --- |
| `vllm/platforms/__init__.py` | 增加 P550 NPU dev platform 的显式 opt-in 检测。 |
| `vllm/platforms/p550_npu.py` | 定义 P550 NPU dev platform，保留 CPU device type，并选择 P550 NPU worker shim。 |
| `vllm/v1/worker/p550_npu_worker.py` | 继承 `CPUWorker`，注册 NPU 单算子框架，不改变 full-model CPU 执行路径。 |
| `vllm/p550/npu/config.py` | 解析环境变量，定义 backend mode、误差阈值、report 路径和 runtime class。 |
| `vllm/p550/npu/runtime.py` | 定义 runtime adapter 协议，提供 CPU fallback runtime 和 unavailable runtime。 |
| `vllm/p550/npu/registry.py` | 算子注册、执行分发、latency 记录、compare 流程、report 写入。 |
| `vllm/p550/npu/golden.py` | 输出 flatten、CPU golden 对比、误差统计、JSONL report 序列化。 |
| `vllm/p550/npu/ops.py` | 内置第一批可验证算子的 CPU golden 实现。 |
| `vllm/p550/npu/cli.py` | 命令行工具，方便本机和板端跑单算子 smoke。 |
| `tests/p550/test_npu_framework.py` | 覆盖算子注册、compare、strict failure、platform opt-in、worker 选择。 |

## 5. 执行模式

核心环境变量：

```bash
export VLLM_P550_NPU_BACKEND=compare
```

支持四种模式：

| 模式 | 行为 | 用途 |
| --- | --- | --- |
| `off` | 不使用 runtime adapter，直接跑 CPU golden。 | 默认安全模式；验证 golden 函数。 |
| `fallback` | 通过 runtime adapter 跑，默认 runtime 是 CPU fallback。 | 验证 adapter/report 路径。 |
| `compare` | runtime 输出与 CPU golden 输出对比。 | 编译器开发的主要模式。 |
| `npu` | 只跑真实 NPU runtime，不做 CPU compare。 | 只建议在 compare 稳定后使用。 |

其他环境变量：

```bash
export VLLM_P550_NPU_OPS=matmul,rms_norm
export VLLM_P550_NPU_ATOL=1e-4
export VLLM_P550_NPU_RTOL=1e-4
export VLLM_P550_NPU_STRICT=1
export VLLM_P550_NPU_REPORT_PATH=/tmp/p550_npu_ops.jsonl
export VLLM_P550_NPU_RUNTIME_CLASS=my_runtime.P550Runtime
```

说明：

- `VLLM_P550_NPU_OPS` 用于限制允许走 NPU 框架的算子。
- `VLLM_P550_NPU_ATOL` 和 `VLLM_P550_NPU_RTOL` 控制数值误差阈值。
- `VLLM_P550_NPU_STRICT=1` 时，compare 失败会直接抛异常。
- `VLLM_P550_NPU_REPORT_PATH` 用于保存 JSONL 记录。
- `VLLM_P550_NPU_RUNTIME_CLASS` 用于注入真实 NPU runtime。

## 6. 执行流程

### 6.1 Platform 启用流程

普通 CPU backend：

```text
未设置 VLLM_P550_NPU_BACKEND
        |
        v
cpu_platform_plugin()
        |
        v
CpuPlatform + CPUWorker
```

P550 NPU dev framework：

```text
VLLM_P550_NPU_BACKEND=compare
        |
        v
p550_npu_platform_plugin()
        |
        v
P550NpuPlatform
        |
        v
P550NpuWorker extends CPUWorker
        |
        v
full-model path still uses CPU backend
single-op path uses vllm.p550.npu
```

### 6.2 单算子 compare 流程

以 `compare` 模式为例：

```text
run_op("matmul", lhs, rhs)
        |
        v
registry 查找 matmul
        |
        v
runtime.run_op("matmul", inputs, attrs)
        |
        v
CPU golden: torch.matmul(lhs, rhs)
        |
        v
compare shape/dtype/value
        |
        v
write JSONL report
        |
        v
return output or raise on strict failure
```

如果当前没有真实 NPU runtime，默认 runtime 是 `CpuFallbackRuntime`，它会调用同一个 CPU golden 函数。这用于验证框架本身是否通畅。

## 7. 当前内置算子

第一批内置算子在 `vllm/p550/npu/ops.py` 中：

| 算子 | 说明 |
| --- | --- |
| `add` | elementwise add，支持 PyTorch broadcasting。 |
| `matmul` | matrix multiply，别名 `linear`。 |
| `silu_mul` | SwiGLU 类 fused helper：`silu(x) * gate`。 |
| `rms_norm` | RMSNorm forward，不含 residual add。 |

这些算子不是最终全集，而是第一批适合建立开发闭环的样例：

- 输入输出是普通 tensor。
- CPU golden 实现清晰。
- shape/dtype 容易检查。
- 后续可扩展到 attention、paged KV、量化 matmul 等更复杂算子。

## 8. 编译器开发人员如何接入

### 8.1 开发一个新算子的基本步骤

1. 在 `vllm/p550/npu/ops.py` 中注册 CPU golden 算子。
2. 在自己的 runtime wrapper 中实现同名 NPU 算子。
3. 设置 `VLLM_P550_NPU_RUNTIME_CLASS` 指向 wrapper class。
4. 用 CLI 跑 `compare`。
5. 检查 JSONL report。
6. 加单元测试或 smoke test。
7. 误差稳定后再考虑接入更高层模型路径。

### 8.2 添加 CPU golden 算子

示例：

```python
@register_op("my_op", description="My P550 NPU candidate op.")
def _my_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, weight)
```

要求：

- golden 函数必须确定性强。
- 输入输出必须是 `torch.Tensor` 或 tensor tuple/list。
- 尽量先用 `float32` 验证，再扩展到 `float16/bfloat16/int8/int4`。
- 每个新增算子都应有最小测试。

### 8.3 实现真实 NPU runtime wrapper

runtime class 需要实现：

```python
class P550Runtime:
    name = "p550_real_runtime"

    def run_op(self, op_name: str, inputs: tuple[torch.Tensor, ...], attrs: dict):
        if op_name == "matmul":
            return self.run_matmul(inputs, attrs)
        raise NotImplementedError(op_name)
```

然后设置：

```bash
export VLLM_P550_NPU_BACKEND=compare
export VLLM_P550_NPU_RUNTIME_CLASS=my_runtime.P550Runtime
export VLLM_P550_NPU_REPORT_PATH=/tmp/p550_npu_ops.jsonl
```

运行：

```bash
python -m vllm.p550.npu.cli run \
  --op matmul \
  --shape 2x4 \
  --rhs-shape 4x3 \
  --dtype float32 \
  --backend compare \
  --strict \
  --report /tmp/p550_npu_ops.jsonl
```

### 8.4 runtime wrapper 与 EIC7700 API 的关系

本轮在 peter 上观察到的 ESWIN/EIC7700 条件包括：

- `/dev/npu0`
- `eic7700_npu`
- `eic7700_dsp`
- `/usr/include/essdk/es_npu_interface.h`
- `/usr/include/essdk/es_npu_types.h`
- `/usr/lib/libedla_runtime.so`
- `/opt/eswin/bin/es_run_model`

后续真实 runtime wrapper 可以分阶段推进：

1. Python wrapper 调用预编译 `.model` 的外部 runner。
2. Python C extension 或 pybind11 封装 C/C++ runtime。
3. C++ wrapper 直接调用 ESWIN NPU runtime API。
4. 将 wrapper 接回 `VLLM_P550_NPU_RUNTIME_CLASS`。

建议初期边界保持为：

```text
torch.Tensor
  -> contiguous CPU buffer
  -> NPU input buffer
  -> submit NPU task
  -> synchronize
  -> copy output back to torch.Tensor
  -> CPU golden compare
```

不要一开始把 vLLM scheduler、KV cache 和模型结构都拉进来。先把单算子的 I/O 和数值正确性跑稳。

## 9. CLI 使用说明

列出已注册算子：

```bash
python -m vllm.p550.npu.cli list
```

运行 `matmul` compare：

```bash
python -m vllm.p550.npu.cli run \
  --op matmul \
  --shape 2x4 \
  --rhs-shape 4x3 \
  --dtype float32 \
  --backend compare \
  --report /tmp/p550_npu_ops.jsonl
```

运行 `rms_norm` compare：

```bash
python -m vllm.p550.npu.cli run \
  --op rms_norm \
  --shape 2x8 \
  --dtype float32 \
  --backend compare \
  --report /tmp/p550_npu_ops.jsonl
```

输出 JSON 中重点看：

- `ok`
- `op`
- `backend`
- `runtime`
- `runtime_latency_ms`
- `cpu_golden_latency_ms`
- `comparisons[].max_abs_error`
- `comparisons[].max_rel_error`
- `comparisons[].within_tolerance`

## 10. JSONL report 格式

每次 `run_op` 会生成一条 JSONL 记录。典型字段：

```json
{
  "op": "matmul",
  "backend": "compare",
  "runtime": "cpu_fallback",
  "inputs": [
    {"shape": [2, 4], "dtype": "torch.float32", "device": "cpu"},
    {"shape": [4, 3], "dtype": "torch.float32", "device": "cpu"}
  ],
  "outputs": [
    {"shape": [2, 3], "dtype": "torch.float32", "device": "cpu"}
  ],
  "runtime_latency_ms": 13.097585999730654,
  "cpu_golden_latency_ms": 0.030003000119904755,
  "comparisons": [
    {
      "index": 0,
      "shape": [2, 3],
      "dtype": "torch.float32",
      "expected_dtype": "torch.float32",
      "max_abs_error": 0.0,
      "max_rel_error": 0.0,
      "within_tolerance": true,
      "message": ""
    }
  ],
  "ok": true
}
```

这份 report 后续可以用于：

- per-op correctness dashboard
- runtime latency 对比
- dtype/shape coverage 统计
- CI artifact
- 编译器 regression 分析

## 11. P550/peter 执行状态

本轮 peter 状态：

| 项目 | 状态 |
| --- | --- |
| SSH alias | `peter` |
| OS | Ubuntu 24.04.3 LTS |
| Kernel | `6.6.92-2025-eic7700` |
| Architecture | `riscv64` |
| Python | `/home/ubuntu/vllm/.venv-p550/bin/python` |
| Torch | 已存在 |
| NPU device | `/dev/npu0` 存在 |
| NPU kernel module | `eic7700_npu` 已加载 |
| DSP kernel module | `eic7700_dsp` 已加载 |
| NPU header | `/usr/include/essdk/es_npu_interface.h` 存在 |
| NPU runtime lib | `/usr/lib/libedla_runtime.so` 存在 |
| Sample runner | `/opt/eswin/bin/es_run_model` 存在 |
| NPU devfreq | 观察到当前频率 `1500000000` |
| Offline compiler | 未发现 `eaac` 或等价工具 |

peter smoke：

```bash
cd /tmp/vllm_p550_npu_probe
PYTHONPATH=. /home/ubuntu/vllm/.venv-p550/bin/python \
  -m vllm.p550.npu.cli run \
  --op matmul \
  --shape 2x4 \
  --rhs-shape 4x3 \
  --backend compare \
  --report /tmp/p550_npu_probe.jsonl
```

结果：

- `ok=true`
- 输出 shape：`[2, 3]`
- max absolute error：`0.0`
- max relative error：`0.0`

另一次 `add` smoke：

- `ok=true`
- 输出 shape：`[4, 4]`
- max absolute error：`0.0`
- max relative error：`0.0`

说明：

- 这两次是 framework/golden path smoke。
- runtime 使用的是 CPU fallback，不是真实 NPU execution。
- 目的不是证明 NPU kernel 已经执行，而是证明框架能在 peter 的 RISC-V Python/Torch 环境中运行，且不会修改已有 CPU vLLM 部署。

## 12. dude 当前状态

dude 当前未完成验证：

```text
ssh: connect to host 192.168.1.57 port 22: No route to host
```

因此本轮不能确认 dude 上的：

- OS/kernel
- Python/Torch 环境
- NPU runtime 环境
- vLLM CPU backend 当前状态
- 本框架 smoke 结果

如果后续 dude 恢复网络，需要执行与 peter 相同的验证流程。

## 13. 测试情况

本机 WSL 测试命令：

```bash
PYTHONPATH=. /home/zyz/vLLM_deploy/.venv/bin/python \
  -m pytest --confcutdir=tests/p550 tests/p550/test_npu_framework.py -q
```

结果：

```text
5 passed
```

说明：

- 使用 `--confcutdir=tests/p550` 是为了隔离本次小测试。
- repo 顶层 `tests/conftest.py` 依赖额外测试包，例如 `tblib`。
- 本轮目标是验证 P550 NPU framework 本身，不拉起整套 vLLM 测试夹具。

其他验证：

```bash
/home/zyz/vLLM_deploy/.venv/bin/python -m compileall -q \
  vllm/p550 \
  vllm/platforms/p550_npu.py \
  vllm/v1/worker/p550_npu_worker.py \
  tests/p550/test_npu_framework.py
```

```bash
git diff --check
```

均通过。

## 14. 与现有 CPU backend 的边界

这点很关键：本框架不是替换 CPU backend。

当前边界如下：

```text
full-model vLLM inference on P550
        |
        v
existing CPU backend

single-op NPU/compiler development
        |
        v
vllm.p550.npu framework
        |
        v
runtime adapter + CPU golden compare
```

因此：

- 不设置 `VLLM_P550_NPU_BACKEND` 时，NPU dev platform 不启用。
- 设置 `VLLM_P550_NPU_BACKEND` 后，platform shim 仍保持 CPU device type。
- full-model fallback 继续依赖 CPU backend。
- 单算子 NPU 开发必须通过 `run_op`、CLI 或后续明确集成点进入。

这样可以保证实验性 NPU 代码不会污染已有 CPU 推理结论。

## 15. 面向 10 个编译器开发人员的协作建议

建议角色拆分：

| 角色 | 人数 | 主要职责 |
| --- | --- | --- |
| Runtime owner | 1 | 维护 `VLLM_P550_NPU_RUNTIME_CLASS` 对接层，统一 NPU buffer/task API。 |
| Golden owner | 1 | 审查 CPU golden 实现，定义误差阈值和 dtype 策略。 |
| Matmul/Linear owner | 1-2 | 开发 dense matmul、quantized matmul、linear 相关 NPU 算子。 |
| Norm/Activation owner | 1-2 | 开发 RMSNorm、SwiGLU、SiLU/GELU 等算子。 |
| Attention/KV owner | 2 | 研究 attention、KV cache、paged memory 的 NPU 化边界。 |
| Tooling/CI owner | 1 | 整理 CLI matrix、JSONL dashboard、CI smoke。 |
| Integration owner | 1 | 负责后续把稳定单算子接入更高层 vLLM 执行路径。 |

协作规则建议：

- 每个算子一个明确 owner。
- 每个算子必须先有 CPU golden。
- 每个算子必须能单独 CLI compare。
- 每个真实 NPU runtime 接入必须保留 fallback/compare 模式。
- 不允许直接绕过 golden compare 合入真实 NPU 输出路径。
- 不在 platform/worker shim 中堆具体算子逻辑。

## 16. 后续路线

建议分四个阶段推进。

### 阶段 1：框架稳定

目标：

- 固化 `vllm.p550.npu` API。
- 增加更多基础算子 golden。
- 增加 shape/dtype matrix。
- 增加 JSONL 汇总脚本。

候选算子：

- `gelu`
- `softmax`
- `layer_norm`
- `rms_norm_residual`
- `quant_matmul`
- `dequant`
- `rotary_embedding`

### 阶段 2：接入预编译 `.model`

目标：

- 用 peter 上已有 EIC7700 runtime 跑预编译 `.model`。
- 封装最小 runtime wrapper。
- 将 `.model` 输入输出转成 torch tensor。
- 跑 compare。

前置条件：

- 编译器团队提供单算子 `.model`。
- 明确 `.model` 输入输出 tensor layout。
- 明确 dtype 和量化参数。

### 阶段 3：接入真实编译器工具链

目标：

- 拿到 `eaac` 或官方等价 compiler。
- 建立 “ONNX/single-op description -> .model” 的自动化路径。
- 将编译产物与 runtime wrapper 绑定。
- 扩展 CI/smoke。

### 阶段 4：接入 vLLM 执行路径

目标：

- 将稳定算子从 standalone `run_op` 接入模型执行路径。
- 优先选择局部、低风险、易验证的算子。
- 保留 CPU fallback。
- 对每个算子增加 runtime switch 和 correctness guard。

不建议跳过阶段 1-3 直接做阶段 4。

## 17. 当前限制

当前框架的限制：

- 不是完整 NPU backend。
- 当前真实 NPU runtime class 尚未实现。
- 当前没有自动调用 EIC7700 `.model` runner。
- 当前没有发现板上离线编译器。
- 当前没有覆盖 attention/KV cache。
- 当前没有自动 dashboard。
- dude 未验证。

这些限制是刻意暴露的，不应被隐藏。框架的价值是先把开发边界和 correctness 机制建立起来。

## 18. 推荐使用方式

本机开发：

```bash
cd /home/zyz/vLLM_deploy/vllm_p550dev_npu
export PYTHONPATH=.
export VLLM_P550_NPU_BACKEND=compare
python -m vllm.p550.npu.cli list
python -m vllm.p550.npu.cli run --op add --shape 4x4 --backend compare
```

peter 上验证：

```bash
cd /home/ubuntu/vllm
git fetch origin p550dev
git checkout p550dev
export PYTHONPATH=.
export VLLM_P550_NPU_BACKEND=compare
/home/ubuntu/vllm/.venv-p550/bin/python -m vllm.p550.npu.cli list
/home/ubuntu/vllm/.venv-p550/bin/python -m vllm.p550.npu.cli run \
  --op matmul \
  --shape 2x4 \
  --rhs-shape 4x3 \
  --backend compare \
  --report /tmp/p550_npu_ops.jsonl
```

真实 NPU wrapper 验证：

```bash
export VLLM_P550_NPU_BACKEND=compare
export VLLM_P550_NPU_RUNTIME_CLASS=my_runtime.P550Runtime
export VLLM_P550_NPU_STRICT=1
export VLLM_P550_NPU_REPORT_PATH=/tmp/p550_npu_real.jsonl

python -m vllm.p550.npu.cli run \
  --op matmul \
  --shape 128x128 \
  --rhs-shape 128x128 \
  --dtype float32 \
  --backend compare \
  --strict
```

## 19. 本轮工作总结

本轮完成的是 P550 NPU backend 开发的第一层基础设施：

- 它没有冒进地宣称完整 NPU backend 已经完成。
- 它保留了现有 CPU backend 的可运行路径。
- 它为 NPU 单算子开发建立了清晰接口。
- 它提供了 CPU golden compare，适合 compiler bring-up。
- 它已经在本机和 peter 上做过 smoke。
- 它明确记录了 EIC7700 当前能做 runtime 验证，但缺少离线编译器闭环。

这套框架的正确使用方式是：先用它把单算子的数值正确性、runtime 接口、shape/dtype coverage 和 report 体系跑稳，再逐步把稳定算子接入 vLLM 更高层执行路径。

