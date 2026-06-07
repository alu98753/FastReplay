"""
Benchmark: Naive sample() vs sample_indices() pipeline
========================================================

Context:
  After discovering the stale-data bug (head!=0 causes np.asarray(rb)[:n]
  to return wrong data), we implemented sample_indices() in C++ with
  head-aware random index generation.

  This benchmark measures the end-to-end sample() cost AND decomposes it
  into two phases:
    Phase 1: RNG — generate 256 random indices
    Phase 2: FI  — Fancy Indexing (gather 256 scattered elements)

Two pipelines compared:
  OLD (naive, INCORRECT): np.random.randint() + view[inds]
  NEW (fixed, CORRECT):   rb.sample_indices()  + view[inds]
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import time
import fastreplay

# ── Parameters ──────────────────────────────────────────────
BUFFER_SIZE = 1_000_000
BATCH_SIZE  = 256       # SB3 default
NUM_CALLS   = 10_000    # sample() calls per trial
NUM_TRIALS  = 5         # repeat trials

# ── Setup ───────────────────────────────────────────────────
rb = fastreplay.RingBuffer(BUFFER_SIZE)
for i in range(BUFFER_SIZE):
    rb.push(i)

view = np.asarray(rb)[:BUFFER_SIZE]  # zero-copy view of C++ buffer
np_arr = np.arange(BUFFER_SIZE, dtype=np.int32)  # Pure NumPy baseline array


def run_trial():
    """Run one trial. Returns dict of per-call ns for each measurement."""

    # ── OLD pipeline: NumPy Baseline ──────────────────────

    # Phase 1 only: RNG (np)
    t0 = time.perf_counter_ns()
    old_inds_list = [np.random.randint(0, BUFFER_SIZE, size=BATCH_SIZE)
                     for _ in range(NUM_CALLS)]
    old_rng_ns = (time.perf_counter_ns() - t0) / NUM_CALLS

    # Phase 2 only: FI with np-generated indices on np_arr
    t0 = time.perf_counter_ns()
    for inds in old_inds_list:
        _ = np_arr[inds]
    old_fi_ns = (time.perf_counter_ns() - t0) / NUM_CALLS

    old_total_ns = old_rng_ns + old_fi_ns

    # ── NEW pipeline: FastReplay (C++) ──────────────────────

    # Phase 1 only: RNG (C++)
    t0 = time.perf_counter_ns()
    new_inds_list = [rb.sample_indices(BATCH_SIZE)
                     for _ in range(NUM_CALLS)]
    new_rng_ns = (time.perf_counter_ns() - t0) / NUM_CALLS

    # Phase 2 only: FI with C++-generated indices on C++ view
    t0 = time.perf_counter_ns()
    for inds in new_inds_list:
        _ = view[inds]
    new_fi_ns = (time.perf_counter_ns() - t0) / NUM_CALLS

    new_total_ns = new_rng_ns + new_fi_ns

    return {
        'old_rng': old_rng_ns,
        'old_fi':  old_fi_ns,
        'old_total': old_total_ns,
        'new_rng': new_rng_ns,
        'new_fi':  new_fi_ns,
        'new_total': new_total_ns,
    }


# ── Run trials ──────────────────────────────────────────────
print(f"{'='*72}")
print(f"  Sample Pipeline Benchmark")
print(f"  buffer={BUFFER_SIZE:,}  batch={BATCH_SIZE}  "
      f"calls/trial={NUM_CALLS:,}  trials={NUM_TRIALS}")
print(f"{'='*72}")

all_trials = []
for t in range(NUM_TRIALS):
    result = run_trial()
    all_trials.append(result)
    print(f"\n  Trial {t+1}/{NUM_TRIALS}:")
    print(f"    OLD (np.randint):      RNG={result['old_rng']:,.0f} ns  "
          f"FI={result['old_fi']:,.0f} ns  "
          f"Total={result['old_total']:,.0f} ns")
    print(f"    NEW (sample_indices):  RNG={result['new_rng']:,.0f} ns  "
          f"FI={result['new_fi']:,.0f} ns  "
          f"Total={result['new_total']:,.0f} ns")

# ── Summary ─────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  Summary (median of {NUM_TRIALS} trials, ns/call)")
print(f"{'='*72}")

def median_of(key):
    return int(np.median([t[key] for t in all_trials]))

old_rng   = median_of('old_rng')
old_fi    = median_of('old_fi')
old_total = median_of('old_total')
new_rng   = median_of('new_rng')
new_fi    = median_of('new_fi')
new_total = median_of('new_total')

print(f"\n  {'Phase':<30} {'NumPy Baseline':>18} {'FastReplay (C++)':>20} {'Speedup':>10}")
print(f"  {'-'*30} {'-'*18} {'-'*20} {'-'*10}")
print(f"  {'Phase 1: RNG':<30} {old_rng:>15,} ns {new_rng:>17,} ns {old_rng/new_rng:>9.2f}x")
print(f"  {'Phase 2: Fancy Indexing':<30} {old_fi:>15,} ns {new_fi:>17,} ns {old_fi/new_fi:>9.2f}x")
print(f"  {'─'*30} {'─'*18} {'─'*20} {'─'*10}")
print(f"  {'End-to-End Total':<30} {old_total:>15,} ns {new_total:>17,} ns {old_total/new_total:>9.2f}x")

print(f"\n  RNG improvement:  {old_rng-new_rng:+,} ns  ({(1-new_rng/old_rng)*100:.1f}% faster)")
print(f"  FI  difference:   {old_fi-new_fi:+,} ns  (expected ≈ 0, same memory access pattern)")
print(f"  Total improvement: {old_total-new_total:+,} ns  ({(1-new_total/old_total)*100:.1f}% faster)")

# ── Correctness demo ────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  Correctness: Why sample_indices() is necessary")
print(f"{'='*72}")

rb2 = fastreplay.RingBuffer(5)
for v in [10, 20, 30, 40, 50]:
    rb2.push(v)
rb2.pop()  # head moves to 1
rb2.pop()  # head moves to 2
# Valid data: [30, 40, 50] at physical positions [2, 3, 4]

naive_view = np.asarray(rb2)[:3]
correct_indices = rb2.sample_indices(3)

print(f"\n  After push(10,20,30,40,50) then pop() × 2:")
print(f"    head = 2, size = {rb2.size()}, valid data = [30, 40, 50]")
print(f"")
print(f"    NAIVE: np.asarray(rb)[:3]     = {naive_view}  ← WRONG (stale data)")
print(f"    FIXED: sample_indices(3)       = {correct_indices}")

view2 = np.asarray(rb2)
correct_data = view2[correct_indices]
print(f"           view[sample_indices(3)] = {correct_data}  ← CORRECT")
