========================================================================
FastReplay: A Zero-Copy Atomic-Based Ring Buffer for RL
========================================================================

Basic Information
=================

* **Project Name**: FastReplay
* **GitHub Repository**: https://github.com/alu98753/FastReplay
* **Developer**: alu98753 (Tzu-Cheng Huang)

FastReplay is a C++ library providing a single-producer/single-consumer
(SPSC) ring buffer for experience replay in reinforcement learning.
It exposes its internal memory buffer to Python via pybind11's
buffer protocol to achieve zero-copy data sharing between C++ and Python.

System Requirements
===================
* **C++ Standard**: C++11 (For cross-platform compatibility)
* **CMake**: 3.15+ (Recommended)
* **Python**: 3.8+ (For pybind11 integration)

Technical Definition
====================

To ensure clarity and avoid misaligned understandings:

1. This project implements an SPSC queue that ultimately
   uses ``std::atomic`` and explicit memory ordering.
2. It is **not** guaranteed to be "lock-free" in the strict hardware sense
   on all platforms (i.e., ``std::atomic<T>::is_lock_free()`` may return
   false depending on the architecture and type ``T``), but it
   specifically avoids OS-level primitives like ``std::mutex``.
3. **Locking Behavior Specification:** The code avoids explicit
   locking in the hot data path (push/sample). However, the
   underlying ``std::atomic`` implementation may use locks
   internally on some platforms. Memory allocation during
   initialization and Python-side object management may also
   involve underlying OS locks.

Problem to Solve
================

The primary problem is to eliminate mutex contention on the replay data
path and share memory buffers between C++ and Python (NumPy) without
copying.

Multi-dimensional arrays holding RL transitions require fast concurrent
writes from the actor (producer) and fast batch sampling from the learner
(consumer). Standard Python-based multiprocessing or lock-based C++ queues
suffer from locking overhead. Passing sampled mini-batches back to Python
typically results in memory copies.

FastReplay focuses on providing an atomic-based C++ data
structure to solve the writing contention and exposing the
underlying contiguous memory directly to Python via pybind11's
buffer protocol to avoid copying.

Prospective Users
=================

Researchers and engineers building single-node RL pipelines who need
high-throughput data collection without the overhead of heavy distributed
systems (like Reverb) or Python multiprocessing.

System Architecture
===================

FastReplay uses atomic head/tail indices for the SPSC logic. It
allocates a contiguous memory block in C++.

The C++ buffer must outlive any Python views referencing
its memory. This project avoids ``std::shared_ptr``
(which involves internal locking for reference counting).
Lifetime safety will be handled through pybind11's
ownership policies from the start of Python integration.

In Python, the C++ buffer is wrapped using pybind11, exposing
memory blocks directly as NumPy ``ndarray`` via the buffer protocol.

Engineering Infrastructure
==========================

* Build System: CMake.
* Testing: Google Test (C++) for concurrency; ``pytest`` (Python) for
  zero-copy validation and functional correctness.
* CI: GitHub Actions (Linux).
* Analysis: ThreadSanitizer (TSan) to detect data races.

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
  copying. Extend ``pytest`` suite to verify zero-copy.

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
  Harden memory lifetime guarantees and test edge cases
  (e.g., buffer deletion while Python views exist).

Week 7 (05/11 to 05/17):
  Benchmarking: atomic-based SPSC vs. mutex-based baseline,
  measuring throughput and latency under concurrent load.

Week 8 (05/18 to 05/24):
  Buffer week for bug fixes, documentation cleanup, and
  final presentation preparation.

References
==========

* cpprb: https://github.com/ymd-h/cpprb
* modmesh: https://github.com/solvcon/modmesh