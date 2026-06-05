# P550 EIC7702 PowerVR OpenCL GPU Debug Notes

This note records the current code-level investigation for using the
Imagination PowerVR AXM GPU on an EIC7702/P550 board through llama.cpp OpenCL.
It intentionally excludes board addresses, user names, passwords, and model
files.

## Goal

Run a small Qwen2.5 0.5B GGUF model with GPU participation on the PowerVR
OpenCL backend. The stable production path remains vLLM CPU inference; this GPU
work is a separate proof of concept.

## Current Findings

- The OpenCL ICD exposes a `PowerVR A-Series AXM-8-256` device.
- llama.cpp upstream OpenCL rejects this device as unsupported.
- Temporarily allowing PowerVR through device detection makes
  `llama-cli --list-devices` show the GPU.
- The first kernel build failure was caused by missing Intel/QCOM subgroup
  constants:

  ```text
  error: use of undeclared identifier 'N_SIMDGROUP'
  error: use of undeclared identifier 'N_DST'
  error: use of undeclared identifier 'N_SIMDWIDTH'
  ```

- Defining generic Intel-sized constants for PowerVR moves past that build
  failure.
- The second kernel build failure was caused by kernels using `half` without
  enabling `cl_khr_fp16`:

  ```text
  error: loading directly from pointer to type '__global half' requires cl_khr_fp16
  ```

- Adding `#pragma OPENCL EXTENSION cl_khr_fp16 : enable` to the affected kernels
  moves past that build failure.
- Forcing PowerVR through the existing subgroup quantized kernels is not correct:
  F16/Q4_0 returned invalid text in earlier tests and Q4_K_M hung.
- A conservative WIP patch is stored at:

  ```text
  tools/p550_llama_cpp_powervr_opencl_wip.patch
  ```

  It records a minimal PowerVR path that only attempts F16 x F32 matvec using a
  scalar OpenCL kernel and refuses quantized `MUL_MAT` dispatch on PowerVR.

## Current Blocker

The current hard blocker is lower than llama.cpp kernel math: basic PowerVR
OpenCL host/device transfers can hang.

Observed independent self-test results:

```text
platforms=1
device=PowerVR A-Series AXM-8-256
write...
```

The process hangs at blocking `clEnqueueWriteBuffer`.

Another upload-mode test showed:

```text
mode=copy_host_ptr n=1024 bytes=4096
create copy_host_ptr...
read...
```

The process hangs at `clEnqueueReadBuffer` after `CL_MEM_COPY_HOST_PTR` buffer
creation.

SVM capability query succeeds:

```text
svm_err=0 svm=0x1
unified_err=0 unified=1
```

This means the driver reports coarse-grain buffer SVM and unified host memory.
However, a coarse-grain SVM probe still hangs at first host mapping:

```text
device=PowerVR A-Series AXM-8-256
svm=0x1
svm alloc
svm map write
```

The process hangs at `clEnqueueSVMMap(..., CL_MAP_WRITE, ...)`.

These transfer failures explain why `llama-cli -ngl 1` hangs during model
loading even after limiting PowerVR to a single minimal F16 kernel: tensor
upload/readback is not reliable yet.

## Next Steps

1. Reset the board or GPU runtime after a hung OpenCL transfer test.
2. Re-run tiny OpenCL transfer probes before running llama.cpp after any driver
   or kernel update:
   - blocking `clEnqueueWriteBuffer`
   - `CL_MEM_COPY_HOST_PTR` plus readback
   - `CL_MEM_USE_HOST_PTR`
   - `CL_MEM_ALLOC_HOST_PTR` plus map/unmap
   - OpenCL SVM map/unmap
3. If a transfer method works, patch llama.cpp's PowerVR buffer upload/readback
   path to use that method before retesting `llama-cli -ngl 1`.
4. Only after F16 `-ngl 1` answers correctly should Q4_0 or Q4_K_M quantized
   kernels be re-enabled.

## Current Conclusion

PowerVR OpenCL device enumeration and kernel compilation can be moved forward
with local code patches, but useful LLM inference is currently blocked by the
PowerVR OpenCL runtime's host/device transfer behavior on this board image.
