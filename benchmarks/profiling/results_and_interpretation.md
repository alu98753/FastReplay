# FastReplay micro_bench.py Profiling Results and Interpretation

Date: 2026-05-18

Environment:

- Conda env: `FastReplay`
- Python: 3.13.12
- NumPy: 2.4.4
- Script under test: `benchmarks/scripts/micro_bench.py`
- Source logic was not changed.

Profiling tools installed and used in the `FastReplay` conda env:

- `line_profiler` 5.0.2
- `py-spy` 0.4.2
- `scalene` 2.3.0
- `valgrind` 3.26.0
- `linux-perf` 6.3.10

Raw environment/package records:

- `benchmarks/profiling/environment.txt`
- `benchmarks/profiling/conda_list.txt`

## Artifact Layout

- Baseline run: `benchmarks/profiling/baseline/`
- cProfile: `benchmarks/profiling/cprofile/`
- line_profiler: `benchmarks/profiling/line_profiler/`
- Scalene: `benchmarks/profiling/scalene/`
- py-spy: `benchmarks/profiling/py_spy/`
- perf: `benchmarks/profiling/perf/`
- Valgrind/Callgrind: `benchmarks/profiling/valgrind/`
- Profiling harnesses: `benchmarks/profiling/scripts/`

The harness files import existing `micro_bench.py` functions and do not modify
the benchmark implementation:

- `scripts/line_profile_micro_bench.py`
- `scripts/batch_fifo_probe.py`

## Baseline Result

File:

- `baseline/micro_bench_stdout.txt`
- `baseline/micro_bench_time.txt`

Baseline wall time:

- 26.75 s wall clock
- 27.15 s user time
- 0.77 s system time
- Max RSS: 62,320 KB

Key benchmark output:

| Section | FastReplay | NumPy baseline | Interpretation |
|---|---:|---:|---|
| Random sample | 2,858 ns/call | 2,905 ns/call | Almost identical; both mainly use NumPy fancy indexing. |
| Push per element | 6,618 ops/ms Atomic+ZC | 16,190 ops/ms | Python-to-C++ call overhead is visible. |
| Pop per element | 6,900 ops/ms Atomic+ZC | 12,806 ops/ms | Same issue: one pybind11 call per item. |
| Batch `pop_view` | 8,854 MB/s | `np.copy`: 1,530 MB/s | `pop_view` is faster than real copy. |
| Batch `pop_view` vs `arr[:n]` | 8,854 MB/s | `np.view`: 412,074 MB/s | `arr[:n]` is view-only and not a fair data-movement baseline. |

## cProfile Result

Files:

- `cprofile/micro_bench.cprof`
- `cprofile/top_cumtime.txt`
- `cprofile/top_tottime.txt`

Important observations:

- Total: 12,144,438 function calls in 31.371 s.
- `bench_numpy_random_sample`: 15.790 s cumulative.
- `bench_fastreplay_random_sample`: 15.206 s cumulative.
- NumPy reduction/indexing helpers dominate:
  - `numpy.fromnumeric.prod`: 12.112 s cumulative.
  - `numpy.fromnumeric._wrapreduction`: 10.205 s cumulative.
  - `numpy.ufunc.reduce`: 4.546 s internal.

Interpretation:

`cProfile` shows that the full script is dominated by the random-sample section,
not by `pop_view`. Both FastReplay random sample and NumPy random sample spend
their time in NumPy-generated index arrays and fancy indexing. This supports the
claim that the current random-sample benchmark mostly measures NumPy's gather
path, not a native FastReplay `sample()` implementation.

Limitation:

`cProfile` is a Python-level deterministic profiler. It cannot explain detailed
C++ time inside the pybind11 extension.

## line_profiler Result

Files:

- `line_profiler/line_profile.txt`
- `line_profiler/line_profile_time.txt`

Representative line-level timings:

### Random sample

FastReplay random sample:

- `np.asarray(rb)[:buffer_size]`: 25.5 us total, only 0.1%.
- Pre-generating random indices: 45.18 ms, 92.2%.
- `view[inds]`: 2.83 ms total across 1000 calls, 5.8%.

NumPy random sample:

- `np.arange(buffer_size)`: 1.93 ms, 7.5%.
- Pre-generating random indices: 17.57 ms, 67.8%.
- `arr[inds]`: 5.57 ms total across 1000 calls, 21.5%.

Interpretation:

For random sampling, most measured time is not RingBuffer logic. It is random
index generation plus NumPy fancy indexing. The FastReplay path uses a NumPy
view into C++ memory, then still relies on NumPy for gather.

### Per-element push/pop

Two calls were measured for FastReplay push/pop because the harness profiles both
mutex and atomic RingBuffer variants.

- `rb.push(i)`: 1.018 s over 2,000,000 calls, about 509 ns/call.
- Python loop overhead line: 0.625 s over 2,000,002 hits.
- `rb.pop()`: 0.980 s over 2,000,000 calls, about 490 ns/call.
- Python loop overhead line: 0.624 s over 2,000,002 hits.

NumPy per-element:

- `arr[i] = i`: 0.361 s over 1,000,000 calls.
- `_ = arr[i]`: 0.374 s over 1,000,000 calls.

Interpretation:

Per-element RingBuffer operations are slower because each element crosses the
Python/pybind11 boundary. This is a different cost model from pure NumPy array
access inside Python.

### Batch FIFO

- `rb.pop_view(n)`: 0.803 ms.
- `np.copy(arr[:n])`: 6.963 ms.
- `arr[:n]`: 6.146 us, plus `np.arange` setup outside the measured slice line.

Interpretation:

`pop_view` is not slower than a true NumPy copy in this run. It is much slower
than `arr[:n]` because NumPy slicing creates a view and does not move data. That
means comparing `pop_view` against `arr[:n]` answers "how expensive is the API
object/view creation path?" but not "how expensive is data movement?"

## Scalene Result

Files:

- `scalene/micro_bench_scalene.json`
- `scalene/micro_bench_scalene_cli.txt`

Important observations:

- Scalene measured 18.034 s attributed to `micro_bench.py`.
- Large native-time blocks appear at:
  - Line 150: random index generation for FastReplay sample.
  - Line 166: random index generation for NumPy sample.
  - Lines 154-155 and 170-171: loops and fancy indexing.
- `rb.push(i)` and `rb.pop()` are visible as native-time lines, but they are not
  the dominant full-script cost.

Interpretation:

Scalene agrees with `cProfile` and `line_profiler`: the current full script is
mostly exercising NumPy native code around random index generation and fancy
indexing. FastReplay's pybind11 path matters in per-element tests, but not as
the top-level full-script bottleneck.

## py-spy Result

Files:

- `py_spy/micro_bench_speedscope.json`
- `py_spy/micro_bench_raw.txt`

Important raw stack samples:

- FastReplay path samples at `bench_fastreplay_random_sample` line 155.
- NumPy path samples at `bench_numpy_random_sample` line 166 and line 171.
- NumPy internals such as `fromnumeric.prod` / `_wrapreduction` appear in raw
  stacks.

Interpretation:

`py-spy` confirms the same shape from a sampling profiler: the hot stacks are in
random sampling and NumPy indexing. It also produced a speedscope JSON for visual
inspection.

Note:

One `py-spy record --format speedscope` run emitted a child-process detach
warning after producing the output file. A second raw run completed with exit
status 0.

## perf Result

Files:

- `perf/perf_stat.txt`
- `perf/perf_cpu_clock.data`
- `perf/perf_report_stdio.txt`

`perf stat`:

- 21.94 s elapsed for the profiled run.
- 25,128.98 ms task clock.
- 264,447 page faults.
- Hardware events such as cycles, instructions, branches, L1/LLC cache loads
  were reported as `<not supported>`.

Interpretation:

The installed `linux-perf` works, but this WSL/kernel environment does not
provide the usual hardware PMU counters. Therefore, `perf stat` cannot currently
answer cache-miss or instruction-per-cycle questions.

`perf record -e cpu-clock -g`:

- Captured 92,390 software samples.
- Top native symbols include:
  - `mapiter_trivial_get` in NumPy `_multiarray_umath`: 18.42% self.
  - `_PyEval_EvalFrameDefault`: 7.89% self.
  - NumPy random generation symbols:
    - `mt19937_uint32`: 5.33% self.
    - `random_bounded_uint64_fill`: 3.80% self.
  - `malloc` / `cfree` and `PyArray_NewFromDescr_int` also appear.

Interpretation:

The software sampling profile strongly supports that random sample cost is in
NumPy gather/indexing and random index generation. It also shows allocation
costs around creating result arrays during fancy indexing.

## Valgrind / Callgrind Result

Files:

- `valgrind/callgrind.batch_fifo.out`
- `valgrind/callgrind_annotate.txt`
- `valgrind/batch_fifo_probe_stdout.txt`

Focused Batch FIFO probe output under Callgrind:

- `FastReplay pop_view ns`: 20.23 ms under instrumentation.
- `NumPy copy ns`: 15.08 ms under instrumentation.
- `NumPy view ns`: 67.8 us under instrumentation.

Important note:

Callgrind changes absolute timings substantially. Use the instruction profile
and relative hotspots, not the nanosecond numbers, as the primary signal.

Top instruction references:

- Program total: 2.305B instruction references.
- `fastreplay...so`: 14.71% at one extension address.
- Python frame evaluation: 11.72%.
- OpenBLAS thread server: 6.20%.
- Additional `fastreplay...so` addresses: 3.90%, 3.30%, 2.39%, 1.91%, etc.
- Python object/method allocation and lookup appear prominently.

Interpretation:

Callgrind confirms that the focused Batch FIFO path spends meaningful instruction
count inside the FastReplay extension and pybind11/Python dispatch. However,
because the extension lacks debug symbols in this installed build, Callgrind
cannot yet name the exact C++ function or line. For deeper C++ attribution, build
the extension with debug symbols (`-g`) and preferably without stripping symbols.

## C++ Implementation Check

Files inspected:

- `include/fastreplay/ring_buffer.hpp`
- `include/fastreplay/ring_buffer_mutex.hpp`
- `src/fastreplay.cpp`

Capacity behavior:

`RingBuffer(std::size_t capacity)` constructs:

```cpp
capacity_(capacity),
buf_(capacity + 1),
head_(0),
tail_(0)
```

Therefore, the current RingBuffer allocates `capacity + 1` `int` elements during
construction. It is not repeatedly growing during normal `push()` operations.

`pop_view` behavior:

- If the requested range is physically contiguous, it returns a `py::array_t<int>`
  referencing `self.data() + current_head`, then advances the head.
- If the range wraps around the end of the ring buffer, it allocates a new NumPy
  array and copies the two chunks with `memcpy`.

Interpretation:

In the no-wrap case, `pop_view` is intended as zero-copy. It can still have
non-trivial overhead because it crosses pybind11, checks size/head/capacity,
constructs a NumPy array object, attaches `py::cast(self)` as the owner, and
advances the head.

## Final Interpretation

1. The original "pop_view is slower than NumPy slice" concern needs a more exact
   baseline. `arr[:n]` is a view-only operation and moves no data. It is expected
   to be much faster than `pop_view`, because `pop_view` crosses pybind11 and
   constructs a Python/NumPy object around C++ memory.

2. Against a true copy baseline, `pop_view` is faster in the baseline run:
   8,854 MB/s for `pop_view` versus 1,530 MB/s for `np.copy(arr[:n])`.

3. Per-element `push` / `pop` is slower than NumPy because the benchmark makes
   one Python-to-C++ call per item. This measures pybind11 boundary cost as much
   as RingBuffer logic.

4. The random-sample section does not currently measure a native FastReplay
   `sample()` implementation. It measures FastReplay storage exposed as a NumPy
   view plus NumPy fancy indexing. Profilers consistently show NumPy random index
   generation and fancy indexing as the dominant costs.

5. The RingBuffer storage is preallocated with `std::vector<int> buf_(capacity +
   1)`. Slowdown is not explained by vector repeatedly reallocating during push.

6. `perf` hardware counters are not available in this WSL environment, so cache
   miss and IPC claims should not be made from this run. Software sampling still
   points to NumPy gather/indexing and random generation.

7. For exact C++ line-level attribution inside FastReplay, rebuild the extension
   with debug symbols and rerun Callgrind/perf. The current Callgrind output
   shows time in `fastreplay...so`, but not named C++ functions.

## Suggested Meeting Summary

The concrete point to tell the teacher:

> I profiled the current Python micro benchmark without changing its logic. The
> main full-script cost is not `pop_view`; it is the random-sample benchmark,
> especially NumPy random index generation and fancy indexing. For Batch FIFO,
> `pop_view` is slower than `arr[:n]` because `arr[:n]` is only a view, but
> `pop_view` is faster than `np.copy(arr[:n])`, which is the fairer data-movement
> baseline. Also, the RingBuffer vector is preallocated as `capacity + 1`, so the
> slowdown is not caused by vector growth during push. To locate exact C++ line
> costs, the next step is rebuilding the extension with debug symbols and rerun
> Callgrind/perf.

