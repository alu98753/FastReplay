# FastReplay Benchmark

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

## Results

*(To be populated after benchmark execution)*

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
