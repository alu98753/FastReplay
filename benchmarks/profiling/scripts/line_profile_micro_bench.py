"""Line-profiler harness for benchmarks/scripts/micro_bench.py.

This file intentionally does not change micro_bench.py logic. It imports the
existing benchmark functions and profiles representative calls line-by-line.
"""

from pathlib import Path
import sys

import fastreplay
import fastreplay_baseline
from line_profiler import LineProfiler


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "scripts"))

import micro_bench  # noqa: E402


def run_profile():
    profiler = LineProfiler()
    functions = [
        micro_bench.bench_fastreplay_random_sample,
        micro_bench.bench_numpy_random_sample,
        micro_bench.bench_push_per_element,
        micro_bench.bench_pop_per_element,
        micro_bench.numpy_push_per_element,
        micro_bench.numpy_pop_per_element,
        micro_bench.bench_pop_view_batch,
        micro_bench.numpy_copy_batch,
        micro_bench.numpy_view_batch,
    ]
    for func in functions:
        profiler.add_function(func)

    n = micro_bench.ITERATIONS
    capacity = micro_bench.CAPACITY
    batch_size = micro_bench.BATCH_SIZE
    num_samples = micro_bench.NUM_SAMPLES

    micro_bench.warm_up()

    rb_sample = fastreplay.RingBuffer(capacity)
    for i in range(n):
        rb_sample.push(i)

    rb_mutex = fastreplay_baseline.RingBuffer(capacity)
    rb_zc = fastreplay.RingBuffer(capacity)
    rb_batch = fastreplay.RingBuffer(capacity)
    for i in range(n):
        rb_batch.push(i)

    profiled = profiler(
        lambda: (
            micro_bench.bench_fastreplay_random_sample(
                rb_sample, n, batch_size, num_samples
            ),
            micro_bench.bench_numpy_random_sample(n, batch_size, num_samples),
            micro_bench.bench_push_per_element(rb_mutex, n),
            micro_bench.bench_push_per_element(rb_zc, n),
            micro_bench.bench_pop_per_element(rb_mutex, n),
            micro_bench.bench_pop_per_element(rb_zc, n),
            micro_bench.numpy_push_per_element(n),
            micro_bench.numpy_pop_per_element(n),
            micro_bench.bench_pop_view_batch(rb_batch, n),
            micro_bench.numpy_copy_batch(n),
            micro_bench.numpy_view_batch(n),
        )
    )
    profiled()
    profiler.print_stats()


if __name__ == "__main__":
    run_profile()
