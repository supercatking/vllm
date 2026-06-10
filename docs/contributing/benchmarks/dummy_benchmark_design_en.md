# vLLM Dummy Benchmark Design and Implementation Notes (English)

This document explains the dummy benchmark design implemented in the `p550dummy` branch. The mechanism was originally developed on the `benchmark` branch and then migrated onto the `p550dev`-based `p550dummy` branch so it can be used on local hosts, P550/RISC-V boards, and CPU-only accelerator-host environments.

## 1. Goal

The dummy benchmark does not measure model quality, real GPU/NPU kernels, or real CPU model execution. Its purpose is to separate an LLM inference request into two conceptual parts:

1. Host side: vLLM request path, scheduler, batch construction, model-runner invocation, sampled-token return, state updates, and output collection.
2. Device side: prefill/decode computation that would normally run on a GPU, NPU, or another accelerator.

In dummy mode, the device side does not execute real model forward. It does not run attention, GEMM, logits computation, or real sampling. Instead, it sleeps for a configured latency and returns a fixed token id. The CPU still observes a normal prefill/decode completion event, but the compute portion is controlled by the dummy backend.

The benchmark is designed to answer questions such as:

- How much host scheduler/control-plane overhead remains when accelerator compute is modeled as fixed latency?
- What is the lower bound of scheduler/control-plane latency when both dummy prefill and dummy decode latency are set to zero?
- How do TTFT, TPOT, and decode TPS differ across host CPU or SoC platforms?
- On a P550/RISC-V environment that can already run real CPU backend inference, how do we avoid confusing real CPU inference with dummy accelerator benchmarking?

## 2. Two-Layer Structure

The implementation has two layers.

### 2.1 Low-level layer: `vllm bench latency --dummy-gpu-execution`

The low-level layer extends the existing `vllm bench latency` command. It still creates a vLLM engine, submits prompt token ids, and drives the V1 scheduler/model-runner path. The difference is that model execution is intercepted inside the model runner.

Example:

```bash
vllm bench latency \
  --model /path/to/model \
  --load-format dummy \
  --enforce-eager \
  --input-len 1024 \
  --output-len 512 \
  --batch-size 1 \
  --num-iters-warmup 1 \
  --num-iters 5 \
  --dummy-gpu-execution \
  --dummy-prefill-delay-ms 5 \
  --dummy-decode-delay-ms 1 \
  --dummy-token-id 0 \
  --output-json /tmp/dummy_latency.json
```

This layer is responsible for:

- Passing dummy configuration through environment variables.
- Enabling the dummy branch in the V1 GPU/CPU model runner.
- Validating that every completion has exactly `--output-len` tokens.
- Validating that every generated token equals `--dummy-token-id`.
- Collecting step-level latency records.
- Embedding a `dummy_gpu_latency` object in the output JSON.

### 2.2 High-level layer: `vllm bench synthetic`

The high-level layer is a matrix runner. It wraps the low-level latency benchmark, runs input/output length combinations, and produces structured reports.

Example:

```bash
vllm bench synthetic \
  --model /path/to/model \
  --input-lens 32,128,512,1024 \
  --output-lens 64,128,512 \
  --warmup 1 \
  --repeats 5 \
  --dummy-prefill-ms 0 \
  --dummy-decode-ms 0 \
  --dummy-output-token-id 0 \
  --output-dir /tmp/vllm_synthetic_zero
```

This layer is responsible for:

- Running one low-level latency benchmark per matrix cell.
- Saving raw JSON and logs for every case.
- Producing `matrix_results.json`, `matrix_results.csv`, and `analysis.md`.
- Extracting TTFT, TPOT, decode TPS, and host-adjusted E2E latency.
- Preserving failed cases with an explicit `invalid_reason`.

## 3. Metric Semantics

### 3.1 TTFT / prefill latency

In this benchmark, TTFT is the CPU-visible latency of one synthetic prefill step. It includes:

- scheduler output processing in the model runner;
- vLLM state maintenance such as `_update_states()`;
- configured dummy prefill sleep;
- dummy token tensor construction;
- bookkeeping and `ModelRunnerOutput` construction;
- latency record writing.

It does not include real prefill compute. Therefore it should not be interpreted as true prompt-token compute latency.

When dummy prefill latency is set to zero, TTFT approximates the request-level fixed overhead of the host/control-plane prefill path.

### 3.2 TPOT / decode latency

TPOT is derived from decode step records:

```text
TPOT = sum(decode_step_latency_ms) / decode_token_count
```

For `batch_size=1`, each decode step usually returns one token, so TPOT is close to one host decode-loop iteration latency.

### 3.3 decode TPS

```text
decode TPS = decode_token_count / sum(decode_step_latency_seconds)
```

When dummy decode delay is zero, decode TPS is a useful synthetic score for the platform's vLLM host decode loop.

### 3.4 host-adjusted E2E

The synthetic matrix runner computes:

```text
host_adjusted_e2e_ms = avg_e2e_ms - observed_dummy_device_ms
```

The observed dummy device latency is estimated from recorded dummy prefill/decode steps. This is an approximation for comparing host overhead across platforms, not a hardware performance counter.

## 4. Code Changes

### 4.1 `vllm/benchmarks/latency.py`

This file implements the low-level benchmark interface.

New CLI arguments:

- `--dummy-gpu-execution`
- `--dummy-prefill-delay-ms`
- `--dummy-decode-delay-ms`
- `--dummy-token-id`
- `--dummy-latency-output-json`

New environment-variable bridge:

- `BENCH_DUMMY_GPU_EXECUTION`
- `BENCH_DUMMY_GPU_PREFILL_DELAY_MS`
- `BENCH_DUMMY_GPU_DECODE_DELAY_MS`
- `BENCH_DUMMY_GPU_TOKEN_ID`
- `BENCH_DUMMY_GPU_LATENCY_LOG`

This bridge passes CLI settings to workers/model runners without coupling worker code to argparse.

New `_validate_dummy_outputs()`:

- checks completion length against `--output-len`;
- checks every token id against `--dummy-token-id`;
- rejects beam search because this first implementation covers standard generation only.

New `_summarize_dummy_gpu_records()`:

- splits JSONL step records into `prefill`, `prefill_chunk`, and `decode` groups;
- computes TTFT, prefill latency, decode latency, TPOT, and decode TPS;
- embeds the result under `dummy_gpu_latency` in the final output JSON.

Important behavior:

- Without `--dummy-gpu-execution`, normal `vllm bench latency` behavior is unchanged.
- In dummy mode, `args.async_scheduling = False` so the CPU-visible token completion point is unambiguous.

### 4.2 `vllm/v1/worker/gpu_model_runner.py`

This is the core dummy backend implementation.

New `DummyGPUExecutionState` stores:

- scheduler output;
- step type;
- dummy delay;
- CPU start timestamp;
- request count;
- scheduled token count.

New runner fields include:

- `self.dummy_gpu_execution`
- `self.dummy_gpu_prefill_delay_s`
- `self.dummy_gpu_decode_delay_s`
- `self.dummy_gpu_token_id`
- `self.dummy_gpu_latency_log`
- `self.dummy_gpu_execution_state`

New `_classify_dummy_gpu_step()`:

- returns `prefill` when a request has no computed tokens;
- returns `prefill_chunk` when the current schedule contains more than one token;
- otherwise returns `decode`.

New `_execute_dummy_gpu_model()`:

- runs necessary pre-forward state updates;
- rejects unsupported paths such as pooling, speculative decoding, async scheduling, pipeline broadcast, encoder inputs, prompt logprobs, and KV transfer;
- selects prefill or decode delay;
- stores a `DummyGPUExecutionState`;
- returns `None`, causing `sample_tokens()` to complete the dummy step.

New `_sample_dummy_gpu_tokens()`:

- sleeps for the configured dummy delay;
- constructs `sampled_token_ids` filled with the dummy token id;
- calls `_update_states_after_model_execute()` to advance vLLM request state;
- reuses `_bookkeeping_sync()` for output bookkeeping;
- constructs a valid `ModelRunnerOutput`;
- writes one step-level latency record.

Integration points:

- `execute_model()` dispatches to `_execute_dummy_gpu_model()` when dummy execution is enabled.
- `sample_tokens()` dispatches to `_sample_dummy_gpu_tokens()` when a dummy state is pending.

Design rationale:

- The scheduler, input batch, request state, and output construction remain real vLLM paths.
- Only real model forward/logits/sampling compute is replaced.
- Latency uses CPU `time.perf_counter()`, not CUDA events, because the target metric is CPU-visible completion latency.

### 4.3 `vllm/benchmarks/synthetic.py`

This file implements the high-level matrix runner.

Responsibilities:

- parse `--input-lens` and `--output-lens`;
- construct low-level `vllm bench latency --dummy-gpu-execution` commands;
- map high-level `--dummy-prefill-ms`, `--dummy-decode-ms`, and `--dummy-output-token-id` to low-level flags;
- save raw per-case JSON/log files;
- parse `dummy_gpu_latency` from low-level output;
- calculate `avg_e2e_ms`, `ttft_avg_ms`, prefill/decode statistics, `tpot_ms`, `decode_tps`, and `host_adjusted_e2e_ms`;
- write JSON, CSV, and Markdown reports.

Design rationale:

- The matrix runner reuses the low-level latency benchmark instead of duplicating engine invocation logic.
- Each matrix cell runs in a separate process, isolating engine state, KV cache state, and warmup effects.
- Failed cases are recorded explicitly with `success=false` and `invalid_reason`.

### 4.4 `vllm/entrypoints/cli/benchmark/synthetic.py`

Adds `BenchmarkSyntheticSubcommand`, connecting `vllm.benchmarks.synthetic.add_cli_args()` and `main()` to the `vllm bench synthetic` CLI.

### 4.5 `vllm/entrypoints/cli/benchmark/main.py`

Adds lazy import of `vllm.entrypoints.cli.benchmark.synthetic` and builds nested benchmark subcommands only when the user actually invokes `vllm bench`.

This matters on remote boards because they may have only benchmark/runtime dependencies installed, not the complete serving stack.

### 4.6 `vllm/entrypoints/cli/main.py`

Changes top-level CLI import behavior:

- for `vllm bench`, import only benchmark CLI modules;
- for non-benchmark commands, import serve/openai/launch/run_batch modules as before.

This prevents benchmark commands from failing due to unrelated OpenAI serving dependencies in minimal P550/RISC-V environments.

### 4.7 `vllm/v1/worker/cpu_model_runner.py`

Adds CPU-device dummy support:

- when `BENCH_DUMMY_GPU_EXECUTION=1`, `load_model()` returns without loading real weights;
- `get_supported_generation_tasks()` returns `['generate']`;
- `get_supported_tasks()` returns `('generate',)`.

This is necessary because the CPU backend can run real Qwen 0.5B inference on P550, but the dummy benchmark must not turn into a real CPU model benchmark. If model loading is skipped, the normal supported-task path would fail when trying to access `self.model`, so dummy mode declares generation support explicitly.

### 4.8 `vllm/v1/worker/cpu_worker.py`

Adds CPU worker dummy handling:

- `_is_synthetic_dummy_execution()` reads `BENCH_DUMMY_GPU_EXECUTION`;
- dummy mode skips startup CPU memory utilization checks;
- `determine_available_memory()` returns a small KV-cache reservation;
- `compile_or_warm_up_model()` returns zero compilation time and skips real warmup.

This prevents CPU-only benchmark runs from being blocked by real CPU backend memory reservation or model warmup behavior.

### 4.9 `tests/benchmarks/test_latency_cli.py`

Adds unit coverage for:

- dummy latency summary calculation;
- TTFT/decode count/TPOT/decode TPS computation;
- dummy output token validation.

### 4.10 `tests/benchmarks/test_synthetic_cli.py`

Adds unit coverage for:

- parsing comma-separated input length lists;
- extracting and summarizing one case from a low-level latency JSON;
- calculating host-adjusted metrics.

## 5. How to Interpret Zero-Delay Results

When the benchmark is run with:

```bash
--dummy-prefill-ms 0 --dummy-decode-ms 0
```

the dummy backend does not intentionally inject device latency. The measured TTFT and TPOT are primarily from:

- scheduler/control-plane logic;
- Python function calls and object manipulation;
- input batch and request state updates;
- dummy token tensor construction;
- output bookkeeping;
- JSONL latency record writing.

Therefore the zero-delay score is a host scheduler lower-bound score, not model throughput.

For example, the peter run on `p550dummy` reported:

```text
geometric mean decode TPS: 1292.11 synthetic tok/s
average TTFT: 1.363 ms
average TPOT: 0.774 ms/token
```

This means that, under this p550dummy implementation and peter environment, the CPU-visible decode-loop scheduler/control-plane cost is about `0.774 ms/token`. It does not imply real Qwen 0.5B inference at 1292 tokens/s.

## 6. Why TTFT Should Not Simply Be Divided by Input Tokens

The current synthetic backend uses a fixed prefill delay. It does not automatically scale prefill delay by input length. In zero-delay mode, no real prefill compute is executed at all.

Therefore:

```text
prefill_per_token = TTFT / input_tokens
```

is only an arithmetic allocation of request-level fixed host overhead across input tokens. It is not a true per-token prefill compute latency.

To benchmark true per-token prefill latency, use a real backend. To benchmark a controlled linear synthetic prefill model, extend the synthetic backend so that:

```text
prefill_delay = fixed_overhead + input_tokens * per_token_delay
```

The current implementation intentionally starts with fixed prefill/decode delays to isolate host scheduler overhead first.

## 7. Safety Boundaries

- Dummy execution is disabled by default.
- It is enabled only by `--dummy-gpu-execution` or indirectly through `vllm bench synthetic`.
- Serving/OpenAI API paths do not use this dummy backend.
- Unsupported paths are explicitly rejected: speculative decoding, async scheduling, encoder inputs, KV transfer, and related complex paths.
- Output token id and output length are validated so invalid dummy runs do not silently pass.
- CPU backend dummy handling is active only when `BENCH_DUMMY_GPU_EXECUTION=1`, so real CPU backend inference remains separate.

## 8. Conclusion

This change implements a micro benchmark for LLM host scheduler evaluation:

- It preserves the vLLM V1 request/scheduler/model-runner/output path.
- It replaces real accelerator compute with configurable dummy delays.
- It reports prefill and decode latency separately.
- It supports zero-delay host lower-bound scoring.
- It runs on CPU-only P550/RISC-V boards without becoming a real CPU inference benchmark.

The benchmark is therefore useful for evaluating how well different host CPUs or SoCs can drive an accelerator-oriented LLM inference control plane.
