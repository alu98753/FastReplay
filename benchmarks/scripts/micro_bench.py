"""
Micro-benchmark: Small Payload (int = 4 Bytes)
Issue #27: [Benchmark/Micro/Python] Small-payload throughput & latency

Metrics (ref: SPSCQueue, Reverb):
  - ops/ms: operations per millisecond
  - MB/s:   memory bandwidth utilization

Three test sections:
  - Per-element: measures single Python->C++ call overhead
  - Batch FIFO:  measures bulk sequential data transfer (on-policy scenario)
  - Random Sample: measures off-policy experience replay pattern (SB3/Tianshou)
     This is the REAL usage pattern for SAC/DQN/TD3.
     Evidence: SB3 off_policy_algorithm.py:500 (add 1-by-1),
               SB3 sac.py:218 (sample batch_size=256 via random indexing)
"""
import ctypes
from ctypes import sizeof
import fastreplay
import fastreplay_baseline
import time
import numpy as np

ITERATIONS = 1_000_000
CAPACITY = ITERATIONS + 1
TEST_SIZE = sizeof(ctypes.c_int)  # 4 bytes
BATCH_SIZE = 256                  # SB3 SAC default
NUM_SAMPLES = 1000                # number of sample() calls per trial
NUM_TRIALS = 1000                    # repeat for statistical stability


# ============================================================
# Helpers
# ============================================================

def warm_up():
    """Warm up CPU caches and JIT before measurement."""
    rb = fastreplay.RingBuffer(1024)
    for i in range(1000):
        rb.push(i)
    for i in range(1000):
        rb.pop()


def cal_ops_ms(n, duration_ns):
    """Calculate operations per millisecond."""
    return n * 1e6 / duration_ns


def cal_mbps(n, duration_ns):
    """Calculate MB/s throughput.

    n items * sizeof(int) bytes each = total bytes transferred.
    """
    total_bytes = n * TEST_SIZE
    return total_bytes / (duration_ns / 1e9) / 1e6


# ============================================================
# Per-element operations (1M individual calls)
# ============================================================

def bench_push_per_element(rb, n):
    """Push n items one by one. Returns elapsed ns."""
    start = time.perf_counter_ns()
    for i in range(n):
        rb.push(i)
    return time.perf_counter_ns() - start


def bench_pop_per_element(rb, n):
    """Pop n items one by one. Returns elapsed ns."""
    start = time.perf_counter_ns()
    for i in range(n):
        rb.pop()
    return time.perf_counter_ns() - start


def numpy_push_per_element(n):
    """Numpy baseline: assign n values one by one."""
    arr = np.zeros(n, dtype=np.int32)
    start = time.perf_counter_ns()
    for i in range(n):
        arr[i] = i
    return time.perf_counter_ns() - start


def numpy_pop_per_element(n):
    """Numpy baseline: read n values one by one."""
    arr = np.arange(n, dtype=np.int32)
    start = time.perf_counter_ns()
    for i in range(n):
        _ = arr[i]
    return time.perf_counter_ns() - start


# ============================================================
# Batch FIFO operations (single bulk call)
# ============================================================

def bench_pop_view_batch(rb, n):
    """Pop n items via zero-copy pop_view (single C++ call).
    Returns elapsed ns.
    """
    start = time.perf_counter_ns()
    _ = rb.pop_view(n)
    return time.perf_counter_ns() - start


def numpy_copy_batch(n):
    """Numpy baseline: copy n items (actual data movement).
    Fair comparison for pop_view because both move data.
    """
    arr = np.arange(n, dtype=np.int32)
    start = time.perf_counter_ns()
    _ = np.copy(arr[:n])
    return time.perf_counter_ns() - start


def numpy_view_batch(n):
    """Numpy view-only: no data movement, just pointer.
    NOT a fair comparison — shown for reference only.
    """
    arr = np.arange(n, dtype=np.int32)
    start = time.perf_counter_ns()
    _ = arr[:n]
    return time.perf_counter_ns() - start


# ============================================================
# Random Sample (off-policy experience replay)
#
# Simulates the REAL usage pattern in SB3 SAC/DQN/TD3:
#   batch_inds = np.random.randint(0, buffer_size, size=batch_size)
#   batch = observations[batch_inds]   # fancy indexing
#
# FastReplay uses buffer protocol (np.asarray) to expose the
# underlying C++ array, then numpy fancy indexing to sample.
# ============================================================

def bench_fastreplay_random_sample(rb, buffer_size, batch_size, num_calls):
    """Simulate off-policy sample(): random index into buffer via
    buffer protocol view. Does NOT remove data (read-only).
    Returns per_call_ns.
    """
    # Get zero-copy view of underlying C++ buffer via buffer protocol
    view = np.asarray(rb)[:buffer_size]

    # Pre-generate random indices (exclude from timing)
    all_inds = [np.random.randint(0, buffer_size, size=batch_size)
                for _ in range(num_calls)]

    start = time.perf_counter_ns()
    for inds in all_inds:
        _ = view[inds]          # numpy fancy indexing (copies scattered data)
    elapsed = time.perf_counter_ns() - start
    return elapsed // num_calls


def bench_numpy_random_sample(buffer_size, batch_size, num_calls):
    """SB3 baseline: pure numpy pre-allocated array + random indexing.
    This is exactly what SB3 ReplayBuffer.sample() does.
    """
    arr = np.arange(buffer_size, dtype=np.int32)

    all_inds = [np.random.randint(0, buffer_size, size=batch_size)
                for _ in range(num_calls)]

    start = time.perf_counter_ns()
    for inds in all_inds:
        _ = arr[inds]
    elapsed = time.perf_counter_ns() - start
    return elapsed // num_calls


# ============================================================
# Main
# ============================================================

def print_row(label, n, duration_ns):
    """Print a formatted result row."""
    ops = cal_ops_ms(n, duration_ns)
    mbps = cal_mbps(n, duration_ns)
    print(f"  {label:<45s} {ops:>12,.0f} ops/ms  {mbps:>10.2f} MB/s")


def print_sample_row(label, ns_list, batch_size):
    """Print a formatted result row with statistical summary.
    ns_list: list of per-call ns values from multiple trials.
    Reports median (min–max) for robustness against WSL noise.
    """
    ns_arr = np.array(ns_list)
    med = int(np.median(ns_arr))
    lo, hi = int(ns_arr.min()), int(ns_arr.max())
    samples_per_ms = batch_size * 1e6 / med
    mbps = batch_size * TEST_SIZE / (med / 1e9) / 1e6
    print(f"  {label:<40s} {med:>6,} ns/call "
          f"({lo:,}–{hi:,})  "
          f"{samples_per_ms:>10,.0f} samples/ms  {mbps:>8.2f} MB/s")


def main():
    warm_up()

    print(f"=== Micro-benchmark: Small Payload "
          f"(int = {TEST_SIZE} Bytes, N = {ITERATIONS:,}) ===")

    # ----------------------------------------------------------
    # Random Sample (REAL RL usage pattern) — FIRST
    # ----------------------------------------------------------
    print(f"\n--- Random Sample "
          f"(off-policy replay, batch={BATCH_SIZE}, "
          f"x{NUM_SAMPLES} calls, {NUM_TRIALS} trials) ---")
    print(f"  Simulates SB3 SAC: replay_buffer.sample(batch_size=256)")

    # Fill buffer
    rb_sample = fastreplay.RingBuffer(CAPACITY)
    for i in range(ITERATIONS):
        rb_sample.push(i)

    # Collect per-call ns across multiple trials
    fr_results = []
    np_results = []
    for trial in range(NUM_TRIALS):
        fr_results.append(bench_fastreplay_random_sample(
            rb_sample, ITERATIONS, BATCH_SIZE, NUM_SAMPLES))
        np_results.append(bench_numpy_random_sample(
            ITERATIONS, BATCH_SIZE, NUM_SAMPLES))

    print(f"\n  [sample(batch_size={BATCH_SIZE})]")
    print_sample_row("FastReplay (buffer_proto + fancy idx)",
                     fr_results, BATCH_SIZE)
    print_sample_row("Numpy baseline (SB3 pattern)",
                     np_results, BATCH_SIZE)

    # ----------------------------------------------------------
    # Per-element
    # ----------------------------------------------------------
    print("\n--- Per-element (1M individual Python->C++ calls) ---")

    # Push
    print("\n  [Push]")
    rb_mutex = fastreplay_baseline.RingBuffer(CAPACITY)
    print_row("FastReplay (Mutex)",
              ITERATIONS, bench_push_per_element(rb_mutex, ITERATIONS))

    rb_zc = fastreplay.RingBuffer(CAPACITY)
    print_row("FastReplay (Atomic+ZC)",
              ITERATIONS, bench_push_per_element(rb_zc, ITERATIONS))

    print_row("Numpy baseline",
              ITERATIONS, numpy_push_per_element(ITERATIONS))

    # Pop
    print("\n  [Pop]")
    print_row("FastReplay (Mutex)",
              ITERATIONS, bench_pop_per_element(rb_mutex, ITERATIONS))

    print_row("FastReplay (Atomic+ZC)",
              ITERATIONS, bench_pop_per_element(rb_zc, ITERATIONS))

    print_row("Numpy baseline",
              ITERATIONS, numpy_pop_per_element(ITERATIONS))

    # ----------------------------------------------------------
    # Batch FIFO (single call)
    # ----------------------------------------------------------
    print("\n--- Batch FIFO (single bulk call for N items) ---")

    # Refill buffer for batch pop test
    rb_batch = fastreplay.RingBuffer(CAPACITY)
    for i in range(ITERATIONS):
        rb_batch.push(i)

    print("\n  [Batch Pop / Slice]")
    print_row("FastReplay pop_view (zero-copy)",
              ITERATIONS, bench_pop_view_batch(rb_batch, ITERATIONS))

    print_row("Numpy np.copy(arr[:n]) (data copy)",
              ITERATIONS, numpy_copy_batch(ITERATIONS))

    print_row("Numpy arr[:n] (view only, ref)",
              ITERATIONS, numpy_view_batch(ITERATIONS))


if __name__ == "__main__":
    main()