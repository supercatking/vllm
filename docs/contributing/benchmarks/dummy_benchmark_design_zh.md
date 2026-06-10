# vLLM Dummy Benchmark 设计与实现说明（中文）

本文系统说明 `p550dummy` 分支中的 dummy benchmark 设计。该机制来自 `benchmark` 分支，并迁移到以 `p550dev` 为基础的 `p550dummy` 分支，用于在本机、P550/RISC-V 板卡以及 CPU-only 或 accelerator-host 环境中评估 vLLM host scheduler/control-plane 开销。

## 1. 设计目标

dummy benchmark 不测模型质量，不测真实 GPU/NPU kernel，也不测真实 CPU 模型执行。它的核心目标是把 LLM 推理链路拆成两个部分：

1. host 侧：request path、scheduler、batch construction、model runner 调用、token 返回、状态更新和输出收集。
2. device 侧：真实场景中由 GPU、NPU 或其他 accelerator 承担的 prefill/decode 计算。

在 dummy 模式下，device 侧不执行真实 forward，不执行 attention、GEMM、logits 计算或真实 sampling，而是按照配置 sleep 一段固定时间，然后返回固定 token id。这样 CPU 侧仍然看到完整的 prefill/decode 完成事件，但计算部分变成可控的 dummy backend。

这个设计用于回答：

- accelerator 计算被抽象为固定 latency 后，vLLM host scheduler 本身有多少开销？
- prefill/decode dummy latency 都设为 0 时，平台的 scheduler/control-plane latency 下限是多少？
- 不同 CPU/SoC 作为 host 控制面时，TTFT、TPOT、decode TPS 的差异是多少？
- 在 P550/RISC-V 这种已经可以跑真实 CPU backend Qwen 0.5B 的环境里，如何避免把真实 CPU inference 和 dummy accelerator benchmark 混淆？

## 2. 两层 benchmark 架构

实现分为两层：底层 `vllm bench latency --dummy-gpu-execution` 和上层 `vllm bench synthetic`。

### 2.1 底层：`vllm bench latency --dummy-gpu-execution`

底层扩展已有 `vllm bench latency`。它仍然创建 vLLM engine，仍然提交 prompt token ids，仍然驱动 V1 scheduler/model-runner 路径；区别是 model runner 内部拦截真实模型计算。

典型命令：

```bash
vllm bench latency   --model /path/to/model   --load-format dummy   --enforce-eager   --input-len 1024   --output-len 512   --batch-size 1   --num-iters-warmup 1   --num-iters 5   --dummy-gpu-execution   --dummy-prefill-delay-ms 5   --dummy-decode-delay-ms 1   --dummy-token-id 0   --output-json /tmp/dummy_latency.json
```

底层负责：

- 把 dummy 配置写入环境变量。
- 在 V1 GPU/CPU model runner 中开启 dummy 分支。
- 验证输出 token 数量等于 `--output-len`。
- 验证所有输出 token 都等于 `--dummy-token-id`。
- 收集 step-level latency records。
- 在 `--output-json` 中写入 `dummy_gpu_latency` 字段。

### 2.2 上层：`vllm bench synthetic`

上层是 matrix runner。它包装底层 latency benchmark，遍历 input/output token 矩阵，生成结构化结果和 Markdown 报告。

典型命令：

```bash
vllm bench synthetic   --model /path/to/model   --input-lens 32,128,512,1024   --output-lens 64,128,512   --warmup 1   --repeats 5   --dummy-prefill-ms 0   --dummy-decode-ms 0   --dummy-output-token-id 0   --output-dir /tmp/vllm_synthetic_zero
```

上层负责：

- 每个 `(input_len, output_len)` case 启动一个底层 latency benchmark。
- 保存每个 case 的 raw JSON 和 log。
- 生成 `matrix_results.json`、`matrix_results.csv`、`analysis.md`。
- 汇总 TTFT、TPOT、decode TPS、host-adjusted E2E。
- 对失败 case 保留 `success=false` 和 `invalid_reason`，方便远端板卡诊断。

## 3. 指标含义

### 3.1 TTFT / prefill latency

在 dummy benchmark 中，TTFT 表示 CPU 侧观察到的一次 prefill step 完成 latency。它包含：

- scheduler output 进入 model runner 后的处理。
- `_update_states()` 等 vLLM 状态维护。
- 配置的 dummy prefill sleep。
- dummy token tensor 构造。
- bookkeeping 和 `ModelRunnerOutput` 构造。
- latency record 写入。

它不包含真实 prefill 计算，因此不能解释为真实 prompt token 的计算耗时。dummy prefill delay 为 0 时，TTFT 近似代表一次 prefill 请求的 host/control-plane 固定开销。

### 3.2 TPOT / decode latency

TPOT 由 decode step records 计算：

```text
TPOT = sum(decode_step_latency_ms) / decode_token_count
```

`batch_size=1` 时，每个 decode step 通常返回一个 token，因此 TPOT 基本等价于一次 host decode loop 的 CPU-visible latency。

### 3.3 decode TPS

```text
decode TPS = decode_token_count / sum(decode_step_latency_seconds)
```

dummy decode delay 为 0 时，decode TPS 可以作为该平台 vLLM host decode loop 的 synthetic score。

### 3.4 host-adjusted E2E

上层 synthetic runner 计算：

```text
host_adjusted_e2e_ms = avg_e2e_ms - observed_dummy_device_ms
```

其中 observed dummy device latency 来自记录到的 dummy prefill/decode step latency。该值用于跨平台比较 host overhead，是近似值，不是硬件性能计数器。

## 4. 代码修改逐项说明

### 4.1 `vllm/benchmarks/latency.py`

这是底层入口。

新增 CLI 参数：

- `--dummy-gpu-execution`
- `--dummy-prefill-delay-ms`
- `--dummy-decode-delay-ms`
- `--dummy-token-id`
- `--dummy-latency-output-json`

新增环境变量桥接：

- `BENCH_DUMMY_GPU_EXECUTION`
- `BENCH_DUMMY_GPU_PREFILL_DELAY_MS`
- `BENCH_DUMMY_GPU_DECODE_DELAY_MS`
- `BENCH_DUMMY_GPU_TOKEN_ID`
- `BENCH_DUMMY_GPU_LATENCY_LOG`

这些变量把 CLI 配置传给 worker/model runner，避免 worker 直接依赖 argparse。

新增 `_validate_dummy_outputs()`：校验输出长度和 token id，确保所有 token 都是 dummy token，并拒绝 beam search 这种当前未覆盖的路径。

新增 `_summarize_dummy_gpu_records()`：解析 JSONL step records，拆分 `prefill`、`prefill_chunk`、`decode`，计算 TTFT、prefill latency、decode latency、TPOT、decode TPS，并写入最终 JSON 的 `dummy_gpu_latency` 字段。

设计边界：不传 `--dummy-gpu-execution` 时，原始 `vllm bench latency` 行为不变；dummy 模式下关闭 async scheduling，使 CPU 侧 token 完成点更明确。

### 4.2 `vllm/v1/worker/gpu_model_runner.py`

这是 dummy backend 的核心。

新增 `DummyGPUExecutionState`，保存 scheduler output、step type、dummy delay、CPU start time、request 数和 scheduled token 数。

新增字段包括：

- `self.dummy_gpu_execution`
- `self.dummy_gpu_prefill_delay_s`
- `self.dummy_gpu_decode_delay_s`
- `self.dummy_gpu_token_id`
- `self.dummy_gpu_latency_log`
- `self.dummy_gpu_execution_state`

新增 `_classify_dummy_gpu_step()`：如果 request 还没有 computed tokens，判定为 `prefill`；如果本轮 scheduled tokens 大于 1，判定为 `prefill_chunk`；否则判定为 `decode`。

新增 `_execute_dummy_gpu_model()`：执行真实 forward 前必要的状态更新，拒绝当前未支持的复杂路径，选择 prefill/decode delay，保存 dummy state，并返回 `None` 让 `sample_tokens()` 完成 dummy sampling。

新增 `_sample_dummy_gpu_tokens()`：sleep 指定 delay，构造全为 dummy token id 的 `sampled_token_ids`，调用 `_update_states_after_model_execute()` 推进 vLLM 状态，复用 `_bookkeeping_sync()` 构造输出，最后写 step-level latency record。

接入点：`execute_model()` 在 dummy enabled 时进入 `_execute_dummy_gpu_model()`；`sample_tokens()` 在存在 dummy state 时进入 `_sample_dummy_gpu_tokens()`。

关键点：scheduler、input batch、request state、output construction 都保留真实 vLLM 路径；只有真实 model forward/logits/sampling compute 被替换。latency 使用 CPU `time.perf_counter()`，因为目标是 CPU 侧可见 latency，而不是 CUDA event latency。

### 4.3 `vllm/benchmarks/synthetic.py`

这是上层 matrix runner。

它负责解析 input/output lens，组装底层 `vllm bench latency --dummy-gpu-execution` 命令，把 `--dummy-prefill-ms`、`--dummy-decode-ms`、`--dummy-output-token-id` 映射到底层参数，保存 raw case JSON/log，并从底层 `dummy_gpu_latency` 中提取 `avg_e2e_ms`、`ttft_avg_ms`、prefill/decode 统计、`tpot_ms`、`decode_tps`、`host_adjusted_e2e_ms`。

设计上，matrix runner 不直接操作 model runner，而是复用底层 latency benchmark，避免重复实现 engine 调用逻辑。每个 case 独立进程运行，隔离 engine 状态、KV cache 状态和 warmup 影响。

### 4.4 `vllm/entrypoints/cli/benchmark/synthetic.py`

新增 `BenchmarkSyntheticSubcommand`，把 `vllm.benchmarks.synthetic.add_cli_args()` 和 `main()` 接入 `vllm bench synthetic`。

### 4.5 `vllm/entrypoints/cli/benchmark/main.py`

新增 synthetic subcommand 的 lazy import，并且只在实际调用 `vllm bench` 时构建 nested benchmark subcommands。这样远端板卡不需要因为 benchmark 命令而提前导入全部 serving 相关模块。

### 4.6 `vllm/entrypoints/cli/main.py`

修改顶层 CLI 导入策略：如果第一个 positional 参数是 `bench`，只导入 benchmark CLI module；非 benchmark 命令才导入 serve/openai/launch/run_batch 等模块。

原因是 P550/RISC-V 最小环境可能只有 benchmark/runtime 依赖，不一定有完整 OpenAI serving 依赖。该修改不改变 `vllm serve` 行为，只减少 `vllm bench` 的 import surface。

### 4.7 `vllm/v1/worker/cpu_model_runner.py`

CPU backend 上新增 dummy 支持：当 `BENCH_DUMMY_GPU_EXECUTION=1` 时，`load_model()` 直接返回，跳过真实权重加载；`get_supported_generation_tasks()` 返回 `['generate']`；`get_supported_tasks()` 返回 `('generate',)`。

这样做的原因是：P550/peter 已经能用 CPU backend 跑真实 Qwen 0.5B，但 dummy benchmark 不能变成真实 CPU inference benchmark。如果跳过模型加载后仍调用 `get_model()` 判断 supported tasks，会因为 `self.model` 不存在而失败，因此 dummy 模式下显式声明 generation 支持。

### 4.8 `vllm/v1/worker/cpu_worker.py`

CPU worker 新增 dummy execution 判断：`_is_synthetic_dummy_execution()` 读取 `BENCH_DUMMY_GPU_EXECUTION`；dummy 模式跳过启动时 CPU memory utilization 检查；`determine_available_memory()` 返回最小 KV cache reservation；`compile_or_warm_up_model()` 返回零 compilation time 并跳过真实 warmup。

这样可以避免 CPU-only benchmark 被真实 CPU backend 的内存 reservation 或模型 warmup 阻塞。

### 4.9 `tests/benchmarks/test_latency_cli.py`

新增底层 dummy latency 单元测试，覆盖 TTFT、decode count、TPOT、decode TPS 的计算，以及 dummy token 输出校验。

### 4.10 `tests/benchmarks/test_synthetic_cli.py`

新增上层 synthetic runner 单元测试，覆盖 input list parser、case summarization 和 host-adjusted metrics 计算。

## 5. Zero-delay 结果如何解释

当配置：

```bash
--dummy-prefill-ms 0 --dummy-decode-ms 0
```

时，dummy backend 不主动注入 device latency。测得的 TTFT 和 TPOT 主要来自 scheduler/control-plane 逻辑、Python 调用、request state 更新、dummy token tensor 构造、output bookkeeping 和 latency record 写入。

因此 zero-delay score 是 host scheduler lower-bound，不是模型吞吐量。

例如 peter 的 `p550dummy` zero-delay 结果为：

```text
geometric mean decode TPS: 1292.11 synthetic tok/s
average TTFT: 1.363 ms
average TPOT: 0.774 ms/token
```

这表示在当前 peter 环境中，decode loop 的 CPU-visible scheduler/control-plane 开销约为 `0.774 ms/token`。它不表示真实 Qwen 0.5B 可以达到 1292 tokens/s。

## 6. 为什么不能直接把 prefill latency 除以 input tokens

当前 synthetic backend 的 prefill delay 是固定值，不随 input token 数量自动线性增长。zero-delay 模式下甚至没有真实 prefill compute。因此：

```text
prefill_per_token = TTFT / input_tokens
```

只能理解为把固定 host overhead 摊到每个 input token 上的数学平均，不是真实 per-token prefill 计算速度。

如果要测真实 per-token prefill，需要用真实 backend；如果要测可控线性 prefill 模型，需要扩展 synthetic backend，例如：

```text
prefill_delay = fixed_overhead + input_tokens * per_token_delay
```

当前实现刻意保持固定 prefill/decode delay，以便先隔离 host scheduler 开销。

## 7. 安全边界

- dummy 默认不开启。
- 只有显式传 `--dummy-gpu-execution` 或通过 `vllm bench synthetic` 间接开启。
- serving/OpenAI API 路径不使用该 dummy backend。
- speculative decoding、async scheduling、encoder inputs、KV transfer 等复杂路径会被显式拒绝。
- 输出 token id 和输出长度强校验，避免错误路径静默通过。
- CPU backend dummy 分支只在 `BENCH_DUMMY_GPU_EXECUTION=1` 时生效，不影响真实 CPU backend 推理。

## 8. 结论

这个修改实现了一个面向 LLM host scheduler 的 micro benchmark：保留 vLLM V1 request/scheduler/model-runner/output path，用可配置 dummy delay 替换真实 accelerator compute，分阶段报告 prefill/decode latency，支持 zero-delay host lower-bound score，并能在 P550/RISC-V CPU-only 板卡上运行而不变成真实 CPU inference benchmark。
