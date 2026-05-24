"""Focused profiling harness for Batch FIFO operations.

This imports the existing micro_bench.py functions and exercises the same
Batch FIFO operations without changing the benchmark implementation.
"""

from pathlib import Path
import sys

import fastreplay


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "scripts"))

import micro_bench  # noqa: E402


def main():
    n = micro_bench.ITERATIONS
    rb = fastreplay.RingBuffer(micro_bench.CAPACITY)
    for i in range(n):
        rb.push(i)

    print("FastReplay pop_view ns:", micro_bench.bench_pop_view_batch(rb, n))
    print("NumPy copy ns:", micro_bench.numpy_copy_batch(n))
    print("NumPy view ns:", micro_bench.numpy_view_batch(n))


if __name__ == "__main__":
    main()
