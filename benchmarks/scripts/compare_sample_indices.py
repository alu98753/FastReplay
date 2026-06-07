"""
Compare: np.random.randint  vs  rb.sample_indices()
Three scenarios:
  A. np indices (pre-generated, outside timing)   ← current benchmark
  B. np indices (generated INSIDE timing)          ← np RNG cost included
  C. rb.sample_indices (generated INSIDE timing)  ← C++ RNG cost included
  D. rb.sample_indices (pre-generated, outside)   ← isolate only C++ Fancy Indexing
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import time
import fastreplay

BUFFER_SIZE = 1_000_000
BATCH_SIZE  = 256
NUM_CALLS   = 1_000   # calls per trial
NUM_TRIALS  = 100     # trials for statistical stability

# --- Setup: fill buffer completely ---
rb = fastreplay.RingBuffer(BUFFER_SIZE)
for i in range(BUFFER_SIZE):
    rb.push(i)

view = np.asarray(rb)[:BUFFER_SIZE]   # zero-copy view for Fancy Indexing

def median_ns(results):
    arr = np.array(results)
    return int(np.median(arr)), int(arr.min()), int(arr.max())

# ---- Scenario A: np.randint pre-generated (current benchmark) ----
A_results = []
for _ in range(NUM_TRIALS):
    all_inds = [np.random.randint(0, BUFFER_SIZE, size=BATCH_SIZE)
                for _ in range(NUM_CALLS)]
    t0 = time.perf_counter_ns()
    for inds in all_inds:
        _ = view[inds]
    A_results.append((time.perf_counter_ns() - t0) // NUM_CALLS)

# ---- Scenario B: np.randint INSIDE timing ----
B_results = []
for _ in range(NUM_TRIALS):
    t0 = time.perf_counter_ns()
    for _ in range(NUM_CALLS):
        inds = np.random.randint(0, BUFFER_SIZE, size=BATCH_SIZE)
        _ = view[inds]
    B_results.append((time.perf_counter_ns() - t0) // NUM_CALLS)

# ---- Scenario C: rb.sample_indices INSIDE timing ----
C_results = []
for _ in range(NUM_TRIALS):
    t0 = time.perf_counter_ns()
    for _ in range(NUM_CALLS):
        inds = rb.sample_indices(BATCH_SIZE)
        _ = view[inds]
    C_results.append((time.perf_counter_ns() - t0) // NUM_CALLS)

# ---- Scenario D: rb.sample_indices pre-generated (isolate Fancy Indexing only) ----
D_results = []
for _ in range(NUM_TRIALS):
    all_inds = [rb.sample_indices(BATCH_SIZE) for _ in range(NUM_CALLS)]
    t0 = time.perf_counter_ns()
    for inds in all_inds:
        _ = view[inds]
    D_results.append((time.perf_counter_ns() - t0) // NUM_CALLS)

# ---- Only index generation (no Fancy Indexing) ----
E_results = []
for _ in range(NUM_TRIALS):
    t0 = time.perf_counter_ns()
    for _ in range(NUM_CALLS):
        _ = np.random.randint(0, BUFFER_SIZE, size=BATCH_SIZE)
    E_results.append((time.perf_counter_ns() - t0) // NUM_CALLS)

F_results = []
for _ in range(NUM_TRIALS):
    t0 = time.perf_counter_ns()
    for _ in range(NUM_CALLS):
        _ = rb.sample_indices(BATCH_SIZE)
    F_results.append((time.perf_counter_ns() - t0) // NUM_CALLS)

print(f"\n{'='*70}")
print(f"  buffer_size={BUFFER_SIZE:,}  batch_size={BATCH_SIZE}  "
      f"{NUM_CALLS} calls × {NUM_TRIALS} trials")
print(f"{'='*70}")
print(f"  {'Scenario':<52} {'median':>8}  {'min':>8}  {'max':>8}")
print(f"  {'-'*52}  {'-'*8}  {'-'*8}  {'-'*8}")

for label, results in [
    ("A: np.randint (pre-gen) + Fancy Idx  [current]", A_results),
    ("B: np.randint (in-timing) + Fancy Idx",          B_results),
    ("C: sample_indices (in-timing) + Fancy Idx",      C_results),
    ("D: sample_indices (pre-gen) + Fancy Idx",        D_results),
    ("E: np.randint only  (no Fancy Idx)",             E_results),
    ("F: sample_indices only  (no Fancy Idx)",         F_results),
]:
    med, lo, hi = median_ns(results)
    print(f"  {label:<52} {med:>7,}ns  {lo:>7,}ns  {hi:>7,}ns")

print()
med_A = median_ns(A_results)[0]
med_D = median_ns(D_results)[0]
med_E = median_ns(E_results)[0]
med_F = median_ns(F_results)[0]
print(f"  [Index generation only]")
print(f"    np.randint cost  : {med_E:,} ns  (B-A ≈ {median_ns(B_results)[0]-med_A:,} ns)")
print(f"    sample_indices   : {med_F:,} ns  (C-A ≈ {median_ns(C_results)[0]-med_A:,} ns)")
print(f"  [Fancy Indexing baseline (pre-gen)]")
print(f"    with np indices  : {med_A:,} ns")
print(f"    with C++ indices : {med_D:,} ns  (diff = {med_D-med_A:+,} ns)")
