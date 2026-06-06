# P550-Based Boards Memory, DMA, and IMG GPU Benchmark Report

Date: 2026-06-06

## Scope

Two P550-based boards on the local network were benchmarked:

- peter: Ubuntu 4-core board
- dude: Debian 8-core board with 2 NUMA nodes

The goal was to measure:

- CPU peak memory bandwidth
- CPU minimum and large-working-set memory latency
- NUMA local and remote memory effects on dude, the 8-core Debian board
- DMA memcpy availability, bandwidth, and latency where safely measurable
- IMG/PowerVR GPU memory bandwidth and latency

All test artifacts were placed in temporary locations on the boards. No source
repository files were modified for the benchmark itself.

## Board Summary

| Board | OS | CPU topology | NUMA | IMG GPU | Notes |
| --- | --- | ---: | ---: | --- | --- |
| peter | Ubuntu 24.04.3 LTS, riscv64 | 4 cores | 1 node | PowerVR A-Series AXM-8-256 | Also has AMD RX550 present; IMG tests were forced to IMG render node |
| dude | Debian trixie/sid, riscv64 | 8 cores | 2 nodes | PowerVR A-Series AXM-8-256 | NUMA node0 CPUs 0-3, node1 CPUs 4-7 |

## Methodology

### CPU Memory

A temporary C benchmark was compiled on both boards as `/tmp/memprobe`.

It measures:

- Read bandwidth
- Write bandwidth
- Copy bandwidth, counted as read plus write bytes
- Triad bandwidth, counted as two reads plus one write
- Random pointer-chase latency over multiple working-set sizes

Bandwidth was measured with arrays larger than cache. Latency was measured with
dependent randomized pointer chasing.

### DMA

The safe kernel interface available on both boards is Linux `dmatest`.

Important caveat: `dmatest` is designed for DMA correctness validation, not as a
clean peak bandwidth benchmark. It can report throughput, but includes random
offset setup and optional verification overhead. A dedicated DMA userspace API
or a purpose-built kernel module would be needed for rigorous DMA peak bandwidth
and latency.

### IMG GPU

Two paths were used:

- `pvr_memory_test`, the vendor PowerVR memory test tool.
- On peter, an EGL/GLES compute shader test forced to
  `/dev/dri/renderD128`, the IMG render node, to measure SSBO copy bandwidth and
  dependent-load latency.

On dude, the available tool results came from `pvr_memory_test`.

## CPU Results

### Peak CPU Memory Bandwidth

Values are GB/s. Peak read is the highest measured read bandwidth on each board.

| Board | Threads | Read | Write | Copy | Triad |
| --- | ---: | ---: | ---: | ---: | ---: |
| peter | 1 | 7.046 | 9.660 | 9.961 | 9.356 |
| peter | 2 | 14.020 | 12.025 | 15.992 | 14.992 |
| peter | 4 | 21.740 | 7.504 | 15.514 | 16.844 |
| dude | 1 | 9.513 | 9.042 | 8.466 | 8.281 |
| dude | 2 | 18.573 | 11.369 | 15.662 | 14.009 |
| dude | 4 | 21.606 | 10.603 | 14.391 | 14.029 |
| dude | 8 | 10.200 | 11.368 | 8.420 | 13.014 |

Peak observed CPU read bandwidth:

| Board | Peak read bandwidth |
| --- | ---: |
| peter | 21.740 GB/s |
| dude | 21.606 GB/s |

The dude aggregate read bandwidth drops at 8 threads because the test
spans the two NUMA nodes without locality control. Controlled NUMA tests below
show why.

### CPU Memory Latency

Values are ns/load from dependent randomized pointer chasing.

| Working set | peter | dude |
| ---: | ---: | ---: |
| 4 KiB | 6.73 | 6.72 |
| 32 KiB | 6.71 | 6.72 |
| 256 KiB | 21.14 | 16.29 |
| 2 MiB | 30.48 | 125.12 |
| 16 MiB | 160.87 | 164.85 |
| 128 MiB | 203.22 | 205.92 |
| 512 MiB | 238.83 | 245.89 |

Minimum observed CPU latency:

| Board | Minimum latency |
| --- | ---: |
| peter | 6.71 ns |
| dude | 6.72 ns |

Large-working-set DRAM latency:

| Board | 128 MiB latency | 512 MiB latency |
| --- | ---: | ---: |
| peter | 203.22 ns | 238.83 ns |
| dude | 205.92 ns | 245.89 ns |

## Debian NUMA Results

dude has two NUMA nodes:

```text
node 0 cpus: 0 1 2 3
node 1 cpus: 4 5 6 7
node distances:
  node0 -> node0: 10
  node0 -> node1: 100
  node1 -> node0: 100
  node1 -> node1: 10
```

Single-thread tests with CPU and memory binding:

| CPU binding | Memory node | Locality | Read GB/s | Write GB/s | Copy GB/s | Triad GB/s | Latency ns |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| CPU 0 | node 0 | local | 9.835 | 9.623 | 9.598 | 9.196 | 222.68 |
| CPU 0 | node 1 | remote | 1.748 | 1.017 | 1.707 | 1.758 | 1074.57 |
| CPU 4 | node 0 | remote | 1.731 | 1.000 | 1.696 | 1.745 | 1084.50 |
| CPU 4 | node 1 | local | 9.739 | 9.569 | 9.607 | 9.064 | 218.80 |

NUMA conclusion:

- Local NUMA memory bandwidth is about 9.7-9.8 GB/s per thread.
- Remote NUMA bandwidth drops to about 1.7 GB/s.
- Remote NUMA latency is about 1.07-1.08 us, roughly 5x local DRAM latency.

## DMA Results

### Ubuntu 4-Core Board

Available DMA engines:

```text
AON: 518c0000.dma-controller-aon, dma0chan0..15
HSP: 50430000.dma-controller-hsp, dma1chan0..11
```

Safe `dmatest` results from the sub-agent run:

| DMA channel | Test | Result | Reported throughput |
| --- | --- | --- | ---: |
| dma1chan0 HSP | 1 MiB buffer, 32 iterations | 32 tests, 0 failures | 135228 KB/s |
| dma0chan0 AON | 1 MiB buffer, 32 iterations | 32 tests, 0 failures | 143981 KB/s |
| auto-selected channels | mixed | pass on dma0chan0/dma0chan1 | about 80-101 MB/s |

A later 8 MiB all-channel `dmatest` attempt was not useful:

```text
summary 0 tests, 0 failures 0.00 iops 0 KB/s (-12)
```

The `-12` result indicates allocation failure. Therefore the safe DMA result to
quote for Ubuntu is about 135-144 MB/s from 1 MiB `dmatest`, with the caveat
that it is not a peak-bandwidth tool.

### Debian 8-Core Board

Available DMA engines include two dies worth of channels:

```text
dma0chan* / dma1chan*
dma2chan* / dma3chan*
```

Attempting an all-channel `dmatest` run caused dude to stop
responding on SSH. Because this is a system stability issue, no further DMA
tests were run on that board.

| Board | DMA status |
| --- | --- |
| peter | Safe dmatest works on selected channels; about 135-144 MB/s reported |
| dude | DMA engines present, but dmatest attempt made board unreachable; no safe bandwidth number |

## IMG GPU Results

### Vendor `pvr_memory_test`

Values are MB/s. These are vendor tool memory copy/set tests involving CPU and
PowerVR device memory mappings.

| Board | Test path | memcpy MB/s | memset MB/s |
| --- | --- | ---: | ---: |
| peter | CPU(Cached) -> DEV(Cached) | 4230.2 | 6823.3 |
| peter | DEV(Cached) -> CPU(Cached) | 4580.9 | N/A |
| peter | CPU(Cached) -> DEV(Uncached) | 615.9 | 630.6 |
| peter | CPU(Cached) -> DEV(Write-Combined) | 624.8 | 629.9 |
| dude | CPU(Cached) -> DEV(Cached) | 4178.6 | 7009.4 |
| dude | DEV(Cached) -> CPU(Cached) | 4472.4 | N/A |
| dude | CPU(Cached) -> DEV(Uncached) | 605.4 | 616.0 |
| dude | CPU(Cached) -> DEV(Write-Combined) | 594.1 | 616.3 |

Extended reverse uncached/write-combined paths:

| Board | Test path | MB/s |
| --- | --- | ---: |
| peter | DEV(Uncached) -> CPU(Cached) | 647.7 |
| peter | DEV(Write-Combined) -> CPU(Cached) | 652.2 |
| dude | DEV(Uncached) -> CPU(Cached) | 654.1 |
| dude | DEV(Write-Combined) -> CPU(Cached) | 607.7 |

### IMG GLES Compute, Ubuntu Board

The sub-agent also ran a GLES compute benchmark explicitly on IMG
`/dev/dri/renderD128`.

Device:

```text
Vendor:   Imagination Technologies
Renderer: PowerVR A-Series AXM-8-256
```

Results:

| Metric | Result |
| --- | ---: |
| SSBO copy, 16 MiB x64 | 2.561 GB/s |
| GPU dependent-load latency, 4 KiB | 148.30 ns/load |
| GPU dependent-load latency, 16 KiB | 168.98 ns/load |
| GPU dependent-load latency, 64 KiB | 173.75 ns/load |
| GPU dependent-load latency, 256 KiB | 179.36 ns/load |
| GPU dependent-load latency, 1 MiB | 250.23 ns/load |
| GPU dependent-load latency, 4 MiB | 330.08 ns/load |
| GPU dependent-load latency, 16 MiB | 335.52 ns/load |
| GPU dependent-load latency, 64 MiB | 337.59 ns/load |

Minimum observed IMG GPU latency:

| Board | Method | Minimum latency |
| --- | --- | ---: |
| peter | GLES compute dependent load | 148.30 ns/load |
| dude | Not measured safely | N/A |

## Key Findings

| Category | peter | dude |
| --- | ---: | ---: |
| CPU peak read bandwidth | 21.740 GB/s | 21.606 GB/s |
| CPU min latency | 6.71 ns | 6.72 ns |
| CPU 128 MiB latency | 203.22 ns | 205.92 ns |
| NUMA remote latency | N/A | about 1.08 us |
| IMG CPU->DEV cached memcpy | 4230.2 MB/s | 4178.6 MB/s |
| IMG DEV->CPU cached memcpy | 4580.9 MB/s | 4472.4 MB/s |
| IMG GPU SSBO copy | 2.561 GB/s | N/A |
| IMG GPU min latency | 148.30 ns | N/A |
| DMA safe measured throughput | 135-144 MB/s | Not safely measured |

## Reliability Notes

- CPU tests are reproducible user-space tests and are the most reliable numbers
  in this report.
- Debian NUMA local/remote results are high-confidence because `numactl
  --hardware` confirms two nodes and explicit CPU/memory binding was used.
- `pvr_memory_test` is a vendor tool and is useful for relative device-memory
  path comparison, but it is not necessarily a pure shader memory bandwidth
  benchmark.
- Ubuntu IMG GLES compute results are the best direct GPU memory latency data
  collected.
- DMA numbers should be treated as functional validation plus rough throughput,
  not as true peak DMA bandwidth.
- dude became unreachable after the DMA test attempt. A power cycle
  or reset may be required before further testing.

## Recommended Follow-Up

1. Reboot or power-cycle dude before any further DMA tests.
2. If DMA peak bandwidth is required, write a small kernel module that:
   - requests one explicit DMA channel,
   - allocates coherent or DMA-mapped buffers,
   - measures only submitted transfer completion time,
   - tests one channel at a time before scaling to multiple channels.
3. For IMG GPU on Debian, rerun the same GLES compute benchmark used on Ubuntu,
   explicitly opening `/dev/dri/renderD128`, after the board is back online.
4. Repeat CPU NUMA measurements with larger iteration counts if publication
   quality numbers are needed.
