========================================================================
FastReplay: A C++ Ring Buffer for RL Experience Replay
========================================================================

Basic Information
=================

* **GitHub Repository**: https://github.com/alu98753/FastReplay
* **Developer**: Tzu-Cheng Huang (alu98753)

FastReplay is a C++ single-producer/single-consumer (SPSC) ring buffer
designed for experience replay in reinforcement learning (RL).  It
exposes the internal memory buffer to Python via pybind11's buffer
protocol so that C++ and Python can share data without copying.

The project started from the hypothesis that replacing Python/NumPy data
structures with a C++ ring buffer would accelerate RL training.  After
implementing the buffer and profiling an end-to-end training loop, the
data showed that buffer operations account for only ~6% of total
training time.  The dominant cost is gradient computation and policy
inference, not the buffer.

Problem to Solve
================

Off-policy reinforcement learning algorithms (DQN, SAC, TD3) store past
transitions in a replay buffer and randomly sample mini-batches for
training.  The two core operations are:

1. ``add(transition)`` — write one transition per environment step
2. ``sample(batch_size)`` — randomly select a batch for training

This project set out to test whether a C++ SPSC ring buffer with atomic
synchronization and zero-copy memory sharing could reduce the overhead
of these operations compared to a pure Python/NumPy implementation.

Prospective Users
=================

Researchers and engineers building single-node RL pipelines who want to
understand whether buffer-level optimization is a worthwhile target, or
who need an SPSC ring buffer with pybind11 integration for streaming
workloads (audio, networking, logging).

System Architecture
===================

The C++ ring buffer uses ``std::atomic`` head and tail indices with
explicit ``acquire``/``release`` memory ordering.  It allocates a
contiguous ``std::vector<int>`` of size ``capacity + 1`` at
construction.

.. code-block:: text

   ┌─────────────────────────────────────────────────────┐
   │                    Python (User API)                 │
   │                                                     │
   │   rb = fastreplay.RingBuffer(capacity=100000)        │
   │   rb.push(value)                                    │
   │   batch = rb.pop_view(n)   ← zero-copy NumPy view   │
   │   arr = np.asarray(rb)     ← buffer protocol         │
   └──────────────┬──────────────────────────────────────┘
                  │  pybind11 buffer protocol
                  │  (no memory copy when contiguous)
   ┌──────────────▼──────────────────────────────────────┐
   │              C++ (ring_buffer.hpp)                    │
   │                                                     │
   │   std::vector<int> buf_(capacity + 1);               │
   │   std::atomic<size_t> head_, tail_;                  │
   │                                                     │
   │   push():  memory_order_release on tail_             │
   │   pop():   memory_order_release on head_             │
   │            memory_order_acquire to read counterpart  │
   └─────────────────────────────────────────────────────┘

The implementation avoids OS-level primitives like ``std::mutex`` on the
hot data path.  However, the underlying ``std::atomic`` may or may not
be hardware lock-free depending on the platform
(``std::atomic<T>::is_lock_free()``).

In Python, pybind11 wraps the C++ buffer and exposes it as a
``numpy.ndarray`` via the buffer protocol.  The C++ buffer must outlive
any Python views referencing its memory; lifetime is managed through
pybind11's ownership mechanism (``py::cast(self)`` as the base object).

API Description
===============

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - API
     - Status
     - Description
   * - ``push(value)``
     - ✅
     - Per-element FIFO write (atomic, no mutex on hot path)
   * - ``pop()``
     - ✅
     - Per-element FIFO read and remove
   * - ``pop_view(n)``
     - ✅
     - Batch FIFO read and remove; returns a NumPy view when contiguous,
       copies when the range wraps around
   * - ``np.asarray(rb)``
     - ✅
     - Exposes C++ memory to NumPy via the buffer protocol
   * - ``sample_indices(batch_size)``
     - ✅
     - Returns random valid physical indices for sampling; accounts for
       head position and wrap-around to avoid reading stale data
   * - ``data_ptr()``
     - ✅
     - Returns the raw memory address of the internal buffer for pointer
       identity verification
   * - Template generalization
     - ❌
     - Currently ``int`` only; ``float``, ``uint8`` not yet implemented
       (Issue #22)

Quick Start
===========

.. code-block:: python

   import fastreplay
   import numpy as np

   rb = fastreplay.RingBuffer(capacity=100000)

   for i in range(1000):
       rb.push(i)

   # Zero-copy batch read
   batch = rb.pop_view(256)

   # Buffer protocol — expose as NumPy array
   arr = np.asarray(rb)

   # Random sampling (head-aware)
   indices = rb.sample_indices(64)

As a drop-in replacement for SB3's replay buffer:

.. code-block:: python

   from stable_baselines3 import DQN
   from fastreplay_sb3 import FastReplayBuffer

   model = DQN("MlpPolicy", "CartPole-v1",
               replay_buffer_class=FastReplayBuffer)
   model.learn(total_timesteps=50_000)

Installation
============

Prerequisites:

* CMake 3.15+
* A C++17 compiler (g++ or clang++)
* Python 3.10+
* pybind11, numpy

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/alu98753/FastReplay.git
   cd FastReplay

   # Install Python dependencies
   pip install pybind11 numpy

   # (Optional) For SB3 integration and benchmarks
   pip install stable-baselines3 gymnasium

Build
=====

There are two ways to build the project.

**Option A — CMake (recommended for development)**

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build

This produces ``fastreplay*.so`` (and ``fastreplay_baseline*.so``) inside
the ``build/`` directory.  To use them from Python, either run Python
from the ``build/`` directory or add it to ``PYTHONPATH``:

.. code-block:: bash

   export PYTHONPATH=$(pwd)/build:$PYTHONPATH
   python -c "import fastreplay; print(fastreplay.RingBuffer(8))"

**Option B — pip install (uses scikit-build-core)**

.. code-block:: bash

   pip install .

This builds and installs the ``fastreplay`` module into the current
Python environment.

Test
====

**C++ unit tests** (19 tests, requires ``-DBUILD_TESTING=ON``):

.. code-block:: bash

   cmake -S . -B build -DBUILD_TESTING=ON
   cmake --build build
   cd build && ctest --output-on-failure

Google Test is fetched automatically via CMake ``FetchContent``.

**Python tests** (29 tests):

.. code-block:: bash

   pytest tests/ -v

Covers push/pop correctness, zero-copy pointer identity, mutation
visibility, ``sample_indices`` validity, and SB3 compatibility.

**ThreadSanitizer** (optional, for race detection):

.. code-block:: bash

   cmake -S . -B build-tsan -DENABLE_TSAN=ON -DBUILD_TESTING=ON
   cmake --build build-tsan
   cd build-tsan && ctest

Demo Script
===========

The ``examples/`` directory contains a ready-to-run demo that trains
two DQN agents on CartPole-v1 — one with the default SB3 replay buffer
and one with ``FastReplayBuffer`` — and compares their rewards:

.. code-block:: bash

   python examples/sb3_dqn_demo.py

Expected output (approximately):

.. code-block:: text

   FastReplay SB3 Integration Demo
   ============================================================
     Training DQN with FastReplayBuffer (C++ index)
     ...
     Reward     : 500.0 ± 0.0

     Training DQN with SB3 default ReplayBuffer
     ...
     Reward     : 492.6 ± 32.0

     ✅ Both agents trained successfully.

To run the micro-benchmarks and profiling scripts:

.. code-block:: bash

   # Batch FIFO and per-element benchmarks
   python benchmarks/scripts/micro_bench.py

   # Random sample pipeline comparison
   python benchmarks/scripts/sample_pipeline_bench.py

   # SB3 training loop time decomposition
   python benchmarks/profiling/scripts/sb3_ab_comparison.py --steps 10000 --trials 5

Benchmark Results
=================

Batch FIFO
----------

``pop_view`` returns a NumPy view directly into C++ memory without
copying:

.. list-table::
   :header-rows: 1

   * - Operation
     - MB/s
   * - ``pop_view`` (zero-copy)
     - 16,703
   * - ``np.copy(arr[:n])``
     - 1,342

``pop_view`` is about 12× faster than a full copy baseline.

Random Sample
-------------

After implementing ``sample_indices(batch_size)`` to fix a stale-data
bug when ``head != 0`` (Issue #28, PR #33):

.. list-table::
   :header-rows: 1

   * - Phase
     - NumPy
     - FastReplay
     - Speedup
   * - Generate random index
     - 5,317 ns
     - 4,094 ns
     - 1.30×
   * - Sample from memory
     - 1,496 ns
     - 1,658 ns
     - 0.90×
   * - End-to-End
     - 7,037 ns
     - 5,737 ns
     - 1.23×

The C++ RNG (MT19937) is faster, but memory gathering is slower due to
the physical-to-logical index mapping from ring buffer wrap-around.

SB3 Integration (DQN + CartPole-v1)
------------------------------------

.. list-table::
   :header-rows: 1

   * - Buffer
     - Reward (50k steps)
     - Wall Time
   * - FastReplayBuffer
     - 500.0 ± 0.0
     - 49.44s
   * - SB3 ReplayBuffer
     - 492.6 ± 32.0
     - 47.86s

Both buffers converge to the same reward level.  FastReplay is ~4.3%
slower end-to-end due to pybind11 boundary crossing overhead.

Training Loop Time Decomposition (5 trials, 10k steps)
------------------------------------------------------------

.. list-table::
   :header-rows: 1

   * - Category
     - SB3 Default
     - FastReplay
   * - Gradient Optimization
     - 4.673s (59.0%)
     - 4.788s (57.9%)
   * - Policy Inference
     - 1.632s (20.6%)
     - 1.723s (20.8%)
   * - Environment Step
     - 0.697s (8.8%)
     - 0.731s (8.9%)
   * - Buffer Total
     - 0.466s (5.9%)
     - 0.552s (6.7%)
   * - Total
     - 7.921s
     - 8.264s

Buffer operations are ~5.9% of total training time.  By Amdahl's Law,
even eliminating all buffer overhead would yield at most ~6.4% speedup.

Six profiling tools (cProfile, line_profiler, py-spy, Scalene, perf,
Valgrind Callgrind) point to the same conclusion: the dominant cost in
the sample path is NumPy's random memory access and random number
generation, which is bounded by cache miss behavior in hardware.

Engineering Infrastructure
==========================

* **Build**: CMake.  ``cmake -S . -B build && cmake --build build``
  compiles the C++ library and pybind11 bindings.
* **C++ Tests**: Google Test — 19 tests including concurrent SPSC stress
  tests.
* **Python Tests**: pytest — 29 tests covering push/pop correctness,
  zero-copy pointer identity, ``sample_indices`` validity, and SB3
  compatibility.
* **Race Detection**: ThreadSanitizer (TSan) — verified zero data races
  under concurrent producer/consumer load (``build-tsan/``).
* **CI**: GitHub Actions (Linux).
* **Profiling**: Six tools used across three system levels (Python
  function, OS sampling, instruction-level).

Schedule
========

Week 1 (03/23 to 03/29):
  Project setup: GitHub repository, CMake build system, and
  minimal pybind11 scaffold.

Week 2 (03/30 to 04/05):
  Implement basic SPSC ring buffer in C++ (mutex-based first).
  Create Python binding and write ``pytest`` tests for
  push/pop correctness from the Python side.

Week 3 (04/06 to 04/12):
  Add zero-copy memory sharing via the buffer protocol so
  that Python receives ``numpy.ndarray`` views without
  copying.  Extend ``pytest`` suite to verify zero-copy.

(04/13 to 04/19): Midterm exam week -- no scheduled work.

Week 4 (04/20 to 04/26):
  Redesign synchronization to use ``std::atomic`` with explicit
  memory ordering (``acquire``/``release``) instead of ``std::mutex``.
  Continue validating correctness through existing Python tests.

Week 5 (04/27 to 05/03):
  Concurrent stress tests (Google Test): run producer and
  consumer threads simultaneously under high throughput.
  Use ThreadSanitizer (TSan) for data-race detection.

Week 6 (05/04 to 05/10):
  SB3 integration: ``FastReplayBuffer`` as a drop-in replacement
  for ``stable_baselines3.ReplayBuffer``.  DQN training on
  CartPole-v1 achieves identical reward.

Week 7 (05/11 to 05/17):
  Training loop profiling (Issue #31): coarse and granular
  time decomposition of DQN training using 6 profiling tools.
  Discovered buffer operations are only ~6% of total time.

Week 8 (05/18 to 06/06):
  Implement ``sample_indices(batch_size)`` (Issue #28, PR #33):
  head-aware random index generation to fix stale-data bug.
  Final presentation preparation.

References
==========

* cpprb: https://github.com/ymd-h/cpprb
* modmesh: https://github.com/solvcon/modmesh
* stable-baselines3: https://github.com/DLR-RM/stable-baselines3
* DeepMind Reverb: https://github.com/google-deepmind/reverb
