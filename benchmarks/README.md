# Benchmark

## Overview

This benchmark framework evaluates FastReplay's SPSC ring buffer
performance through a systematic methodology derived from a literature
survey of the above references and 6 reference implementations:
**SPSCQueue** [4], **MPMCQueue** (Rigtorp), **cpprb**,
**Reverb** [2], **EnvPool** [1], and **Tianshou** [3].

The evaluation is structured into two layers (Micro and Macro), each
with specific metrics, payload configurations, and baseline
comparisons.

---

## Layer 1: Micro-benchmark (No GPU Required)

Isolate and evaluate the raw data structure performance of FastReplay's
SPSC ring buffer, independent of neural network inference and training.

### Metrics

| Metric | Source Framework | Definition |
|---|---|---|
| **ops/ms** | SPSCQueue (Rigtorp) | Push+Pop operations per millisecond |
| **ns RTT** | SPSCQueue (Rigtorp) | Round-trip latency via ping-pong pattern |
| **MB/s** | Reverb (DeepMind) | Memory bandwidth utilization |

*   **ops/ms** directly measures queue logic overhead. SPSCQueue [4]
    reports 362,723 ops/ms on AMD Ryzen 9 3900X
    (vs boost::lockfree::spsc 209,877 ops/ms).
*   **MB/s** identifies whether performance is bounded by queue logic
    or memory bandwidth. Reverb [2] observed an 11 GB/s ceiling across
    all payload sizes (Section 5.3), attributable to network bandwidth
    constraints rather than Reverb itself.
*   **ns RTT** quantifies the cost of memory barriers
    (`memory_order_release/acquire`) via a two-queue ping-pong pattern
    between producer and consumer threads. SPSCQueue [4] reports
    133 ns RTT on the same hardware.

### Payload Levels

Following Reverb's cross-magnitude methodology (400B–400KB), each
payload level maps to a real RL environment for practical relevance:

| Level | RL Environment | Obs Dimension | Transition Size |
|---|---|---|---|
| **Small** | MuJoCo Ant | 111 floats | ~500 B |
| **Medium** | MuJoCo Humanoid | 376 floats | ~1.5 KB |
| **Large** | Atari (preprocessed) | 84×84×4 uint8 | ~28 KB |
| **XL** | High-res Robotics | 256×256×3 uint8 | ~192 KB |

Cross-magnitude testing distinguishes queue logic overhead (visible at
Small) from memory bandwidth saturation (visible at XL).

### Dual-Language Strategy

| Language | Purpose | Reference |
|---|---|---|
| **Python** (primary) | Measures the user-facing API performance including pybind11 binding and zero-copy overhead | EnvPool, cpprb |
| **C++** (diagnostic) | Isolates pure queue logic overhead for root-cause analysis | SPSCQueue (Rigtorp) |

Python-first because FastReplay's target users are RL researchers who
interact exclusively through the Python API. If Python-side benchmarks
show no improvement, C++-side gains are irrelevant to the end user.

### Baselines

1.  **FastReplay (Mutex)**: Current `ring_buffer.hpp` with
    `std::lock_guard<std::mutex>` on all operations
2.  **FastReplay (Mutex + Zero-copy)**: Mutex synchronization with
    `pop_view()` zero-copy path via pybind11 buffer protocol
    (current `fastreplay.cpp` implementation)
3.  **FastReplay (Atomic)**: After Issue #11 (`std::atomic` with
    `memory_order_release/acquire`, `alignas(64)` false-sharing
    prevention)
4.  **FastReplay (Atomic + Zero-copy)**: Full optimized pipeline
5.  **Pure Numpy baseline**: Python-native pre-allocated array with
    index tracking (represents the simplest possible implementation)

---

## Layer 2: Macro-benchmark (GPU Required, RTX 4090)

End-to-end training integration following EnvPool's [1] time
decomposition methodology (Section 4.2, Figure 4).

### Time Decomposition (5 Components)

Adapted from EnvPool's 4-component model for buffer-focused analysis:

1.  **Buffer\_Push\_Time**: `buffer.push(transition)`
2.  **Buffer\_Sample\_Time**: `buffer.sample(batch_size)`
3.  **NN\_Forward\_Time**: Policy network inference
4.  **NN\_Backward\_Time**: Gradient computation and parameter update
5.  **Other\_Time**: Environment step + GPU↔CPU data transfer

### Metrics

| Metric | Source Framework | Definition |
|---|---|---|
| **Buffer Time Proportion** | EnvPool (adapted) | `(Push + Sample) / Total_Time` |
| **SPS** | EnvPool | Steps Per Second for full training loop |
| **Episodic Return vs Runtime** | Tianshou + EnvPool | Learning curve vs wall-clock time |

### Test Configuration

*   **Algorithm**: SAC (Off-policy, high replay ratio) to maximize
    buffer access frequency
*   **SPI Ratio**: Leveraging Reverb's Sample-to-Insert Ratio concept,
    set artificially high SPI and large Batch Size (1024+) to shift the
    bottleneck to the buffer data supply
*   **Environment**: MuJoCo continuous control tasks

---

## Issue Tracking

*   **Parent**: [#23 Comprehensive Performance Evaluation Framework](https://github.com/alu98753/FastReplay/issues/23)
*   **Sub-issues**:
    *   [#27 [Micro/Python] Small-payload throughput & latency](https://github.com/alu98753/FastReplay/issues/27) — **this week**
    *   [#26 [Micro/C++] Pure queue throughput & RTT](https://github.com/alu98753/FastReplay/issues/26)
    *   [#25 [Micro] Cross-payload (Medium/Large/XL)](https://github.com/alu98753/FastReplay/issues/25) — after template generalization
    *   [#24 [Macro] End-to-end time decomposition](https://github.com/alu98753/FastReplay/issues/24) — requires RTX 4090
*   **Related (discovered via benchmark)**:
    *   [#22 DictBuffer + Template generalization](https://github.com/alu98753/FastReplay/issues/22)
    *   **NEW**: `[Feature] C++ sample(batch_size)` — correctness & usability (see Insight 1)

## Current Implementation Status

> **Note**: FastReplay does not yet implement `sample()` or `push_batch()`.
> See [project README](../README.rst) for current API status.
> I simulate `sample()` in `benchmarks/scripts/micro_bench.py` using
> buffer protocol + numpy fancy indexing as a proxy measurement.

## Results

### 1. Small-payload throughput & latency (Issue #27)

Script: `benchmarks/scripts/micro_bench.py`
Environment: WSL2, 1000 trials, reporting median (min–max).

#### Insight 1: Random Sample(with 1000 trials) — No significant difference

Simulates the real off-policy RL pattern
(SB3 `sac.py:218`: `replay_buffer.sample(batch_size=256)`).

**How this test works (important context)**:
FastReplay does not yet have a `sample()` method. To simulate it,
I use the existing buffer protocol to let numpy read directly from
FastReplay's C++ memory:

1. **Storage**: FastReplay's C++ `std::vector` (data lives in C++)
2. **Reading**: `np.asarray(rb)` creates a zero-copy view into C++
   memory, then numpy fancy indexing gathers 256 random elements.

So the "FastReplay" row below measures: **FastReplay storage + numpy reading**.
The "Numpy baseline" row measures: **numpy storage + numpy reading**.

| Implementation | Median (ns/call) | Range | samples/ms |
|---|---:|---|---:|
| FastReplay (C++ storage + numpy read) | 2,033 | 1,042–11,538 | 125,922 |
| Numpy baseline (numpy storage + numpy read) | 1,941 | 609–5,711 | 131,891 |

**Difference: ~5%, within measurement noise. Ranges overlap heavily.**

Why? Both use the same numpy fancy indexing to gather data. The
bottleneck is **memory scatter-gather** — the CPU must visit 256
random memory locations regardless of who owns the memory.

**This motivates implementing C++ `sample()`**: a native C++ sample()
would handle both storage AND reading internally, eliminating numpy
from the read path entirely. More critically, the current
`np.asarray(rb)[:size]` approach has a **correctness issue**: when
`head ≠ 0` (after pops), it returns stale data from popped positions,
because the physical layout no longer matches the logical data order.
For example:

```
After push(10,20,30,40,50) then pop() twice:
  Physical memory: [10, 20, 30, 40, 50]   head=2, size=3
  Valid data:               [30, 40, 50]   (positions 2,3,4)
  np.asarray(rb)[:3] returns [10, 20, 30]  ← WRONG (includes popped 10, 20)
  C++ sample() would know to start at head=2 and return from [30, 40, 50]
```

C++ `sample()` is needed for correctness and usability, not just
performance.

---

#### Insight 2: Per-element — pybind11 call overhead dominates

| Operation | FastReplay (Mutex) | Numpy | Ratio |
|---|---:|---:|---|
| Push | ~9,500 ops/ms | ~23,000 ops/ms | Numpy 2.4x faster |
| Pop | ~9,500 ops/ms | ~22,000 ops/ms | Numpy 2.3x faster |

pybind11's per-call overhead ≈ **50 ns** fixed cost. Over 1M calls
this accumulates significantly. In batch operations (Random Sample,
Batch FIFO), this cost is amortized to near zero because only 1
cross-boundary call is made.

**Conclusion**: While FastReplay's `push()` is slower than pure numpy
due to pybind11 overhead, this overhead (~50ns) is completely
negligible in real RL loops, where `env.step()` takes 0.08–5 ms
depending on environment complexity [1, Section 4.2].
Therefore, optimizing per-element push or adding a `push_batch()`
API is a **low-priority "nice-to-have"**, not a critical bottleneck.

---

#### Insight 3: Batch FIFO — pop_view zero-copy achieves 16.7 GB/s

| Operation | MB/s | Note |
|---|---:|---|
| pop_view (zero-copy) | 16,703 | Near DDR4 bandwidth ceiling |
| np.copy (data copy) | 1,342 | Requires allocation + memcpy |
| np.view (ref only) | 521,444 | No data movement (misleading) |

pop_view is **12.4x faster** than np.copy. This validates the
zero-copy path's value for sequential bulk operations (on-policy
rollout consumption).

Note: `np.view` does no data movement at all (just creates a pointer
object), so its MB/s figure is not a meaningful bandwidth measure.

---

### Architectural implications

These results inform the development roadmap:

1.  **C++ `sample(batch_size)` is required** (new Issue [#28]
    (https://github.com/alu98753/FastReplay/issues/28),
    depends on #22):
    Not for performance (bottleneck is memory scatter-gather),
    but for correctness (head offset handling) and usability
    (matching SB3/Tianshou API expectations).
2.  **Batch push API is low priority**:
    Although 1M individual pybind11 calls are slow, in a real RL
    loop, `push()` is called once per `env.step()`. The ~50ns overhead
    is negligible compared to env.step time (0.08–5 ms) [1, Section 4.2].
3.  **Atomic migration (Issue #11)** will not show improvement in
    single-threaded random sample. Its value lies in enabling
    true SPSC concurrent push/sample without lock contention.



## References

### Academic Papers
*   [1] Weng, J. et al. (2022). "EnvPool: A Highly Parallel
    Reinforcement Learning Environment Execution Engine."
    *NeurIPS 2022, Datasets and Benchmarks Track.*
    [arXiv:2206.10558](https://arxiv.org/abs/2206.10558)
    — Local: `benchmarks/literature/envpool.txt`
*   [2] Cassirer, A. et al. (2021). "Reverb: A Framework for
    Experience Replay." *arXiv preprint.*
    [arXiv:2102.04736](https://arxiv.org/abs/2102.04736)
    — Local: `benchmarks/literature/reverb.txt`
*   [3] Weng, J. et al. (2022). "Tianshou: A Highly Modularized Deep
    Reinforcement Learning Library."
    *Journal of Machine Learning Research (JMLR), 23, 1-6.*
    [arXiv:2107.14171](https://arxiv.org/abs/2107.14171)
    — Local: `benchmarks/literature/tianshou.txt`

### Reference Implementations
*   [4] Rigtorp, E. SPSCQueue: Wait-free and lock-free SPSC queue.
    [GitHub](https://github.com/rigtorp/SPSCQueue)
    — Local benchmark: `lib/SPSCQueue/src/SPSCQueueBenchmark.cpp`
*   EnvPool benchmark script:
    `lib/envpool/benchmark/test_envpool.py`

## Tools

*   NotebookLM: https://notebooklm.google.com/notebook/79483479-0e39-4b45-afeb-e136dc0688b5
