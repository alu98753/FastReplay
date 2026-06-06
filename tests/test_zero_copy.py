"""Zero-copy Buffer Protocol validation."""
import pytest
import numpy as np
import fastreplay

def test_buffer_protocol_base():
    """valid balic Buffer protocol and can be read by numpy"""
    buf = fastreplay.RingBuffer(4)
    buf.push(10)
    buf.push(20)

    arr = np.asarray(buf)

    assert isinstance(arr, np.ndarray)
    # arr has base and same content as buf
    assert arr.base is not None 
    assert arr.shape == (5,) # physical capacity is capacity + 1
    assert arr[0] == 10
    assert arr[1] == 20



def test_zero_copy_pop_view_contiguous():
    """Test pop_view returns a zero-copy view for contiguous data."""
    buf = fastreplay.RingBuffer(4)
    for i in range(4):
        buf.push(i)
    
    base_arr = np.asarray(buf)
    view_arr = buf.pop_view(3)

    assert view_arr.shape == (3,)
    assert view_arr.strides == base_arr.strides
    assert view_arr[0] == 0
    assert view_arr[2] == 2

    assert np.shares_memory(base_arr, view_arr)

def test_zero_copy_pop_view_wrap_around():
    buf = fastreplay.RingBuffer(4)
    for i in range(4):
        buf.push(i*10)
    buf.pop()
    buf.pop()
    buf.push(50)
    buf.push(60)

    # wrap around
    base_arr = np.asarray(buf)
    view_arr = buf.pop_view(4)

    assert view_arr.tolist() == [20, 30, 50, 60]
    assert np.shares_memory(base_arr, view_arr) is False

def test_pop_view_insufficient_data():
    buf = fastreplay.RingBuffer(4)
    buf.push(10)
    buf.push(20)

    with pytest.raises(RuntimeError, match="Not enough data"):
        buf.pop_view(3)


#TODO : Dangling pointer


def test_head_nonzero_stale_data():
    """Demonstrate that np.asarray(rb)[:size] reads STALE data when head != 0.

    This is the correctness bug discovered in Issue #27 / Slide 6:
    After push(10,20,30,40,50) then pop() twice, head advances to index 2.
    The logical buffer content is [30, 40, 50], but a naive
    np.asarray(rb)[:rb.size()] returns [10, 20, 30] — the first 3 PHYSICAL
    elements, which includes already-popped stale data (10, 20) and misses
    valid data (40, 50).

    This test PROVES the bug exists. Fixing it requires a native C++ sample()
    that is aware of the ring buffer's head position (see Issue #28).
    """
    rb = fastreplay.RingBuffer(5)
    for v in [10, 20, 30, 40, 50]:
        rb.push(v)

    # Pop two elements — head moves from 0 to 2
    assert rb.pop() == 10
    assert rb.pop() == 20
    assert rb.size() == 3  # logical content: [30, 40, 50]

    # The naive approach: treat the underlying memory as a flat array
    arr = np.asarray(rb)
    naive_slice = arr[:rb.size()]  # takes first 3 physical slots

    # BUG: naive_slice contains stale popped data [10, 20, 30]
    # instead of the correct logical content [30, 40, 50]
    assert naive_slice.tolist() == [10, 20, 30], (
        "Expected the naive slice to show the BUG: stale physical data"
    )

    # The correct logical content starts at physical index 2 (= head)
    correct_data = [arr[2], arr[3], arr[4]]
    assert correct_data == [30, 40, 50], (
        "Correct data must start from head position, not index 0"
    )


# ---- sample_indices tests (Issue #28 fix) ----

def test_sample_indices_fixes_stale_data():
    """sample_indices returns head-aware indices that avoid stale data.

    This is the COMPLEMENT to test_head_nonzero_stale_data above:
    that test PROVES the bug exists with naive slicing;
    this test PROVES sample_indices fixes it.
    """
    rb = fastreplay.RingBuffer(5)
    for v in [10, 20, 30, 40, 50]:
        rb.push(v)

    rb.pop()  # discard 10, head → 1
    rb.pop()  # discard 20, head → 2
    assert rb.size() == 3

    arr = np.asarray(rb)

    # Use sample_indices to get valid physical indices
    indices = rb.sample_indices(1000)

    # Every sampled value must be from {30, 40, 50}, never {10, 20}
    sampled_values = set(arr[indices].tolist())
    assert sampled_values <= {30, 40, 50}, (
        f"sample_indices returned stale data: {sampled_values - {30, 40, 50}}"
    )
    # With 1000 samples from 3 elements, we should hit all 3
    assert sampled_values == {30, 40, 50}, (
        f"Expected all valid values, got {sampled_values}"
    )


def test_sample_indices_wrap_around():
    """sample_indices handles wrap-around correctly."""
    rb = fastreplay.RingBuffer(4)  # phys capacity = 5
    for i in range(4):
        rb.push(i * 10)
    rb.pop()  # head=1
    rb.pop()  # head=2
    rb.pop()  # head=3
    rb.push(100)  # slot 4
    rb.push(110)  # slot 0 (wrapped)
    # valid slots: {3, 4, 0}, values: {30, 100, 110}
    assert rb.size() == 3

    arr = np.asarray(rb)
    indices = rb.sample_indices(500)

    sampled_values = set(arr[indices].tolist())
    assert sampled_values == {30, 100, 110}, (
        f"Expected {{30, 100, 110}}, got {sampled_values}"
    )


def test_sample_indices_empty_raises():
    """sample_indices on empty buffer raises RuntimeError."""
    rb = fastreplay.RingBuffer(4)
    with pytest.raises(RuntimeError, match="empty"):
        rb.sample_indices(1)


def test_sample_indices_returns_int64():
    """Returned array dtype must be int64 for NumPy indexing compatibility."""
    rb = fastreplay.RingBuffer(8)
    for i in range(5):
        rb.push(i)

    indices = rb.sample_indices(3)
    assert indices.dtype == np.int64, (
        f"Expected int64, got {indices.dtype}"
    )


# ---- Zero-Copy Semantic Verification (Issue #28 / Slide 5) ----

def test_zero_copy_pointer_identity():
    """Verify zero-copy by checking that the NumPy view's memory address
    matches the C++ ring buffer's internal data pointer.

    This is a hardware-independent verification: if two arrays share the
    same base address, no memory copy occurred. This proves zero-copy as
    a memory semantics guarantee, not a performance claim.
    """
    buf = fastreplay.RingBuffer(8)
    for i in range(5):
        buf.push(i * 10)

    # Get the C++ buffer's raw data pointer (physical index 0)
    cpp_base_ptr = buf.data_ptr()

    # Get a NumPy view via buffer protocol
    arr = np.asarray(buf)
    numpy_base_ptr = arr.ctypes.data

    # Pointer identity: both point to the same memory
    assert cpp_base_ptr == numpy_base_ptr, (
        f"Pointer mismatch: C++ data_ptr=0x{cpp_base_ptr:x}, "
        f"NumPy ctypes.data=0x{numpy_base_ptr:x}. "
        f"This means np.asarray(buf) performed a copy!"
    )


def test_zero_copy_pointer_identity_pop_view():
    """Verify pop_view returns a pointer into the original buffer (no copy)
    when data is contiguous (no wrap-around).

    The returned array's base address should equal
    data_ptr + head * sizeof(int).
    """
    buf = fastreplay.RingBuffer(8)
    for i in range(6):
        buf.push(i * 10)

    # Pop 2 elements so head advances to physical index 2
    buf.pop()
    buf.pop()

    cpp_base_ptr = buf.data_ptr()
    head = 2  # after 2 pops
    expected_view_ptr = cpp_base_ptr + head * np.dtype(np.int32).itemsize

    view = buf.pop_view(4)
    view_ptr = view.ctypes.data

    assert view_ptr == expected_view_ptr, (
        f"pop_view pointer mismatch: expected 0x{expected_view_ptr:x} "
        f"(data_ptr + head*4), got 0x{view_ptr:x}. "
        f"pop_view may have copied data instead of returning a view."
    )
    assert view.tolist() == [20, 30, 40, 50]


def test_zero_copy_mutation_visibility():
    """Verify zero-copy by writing through the NumPy view and confirming
    the mutation is visible when reading back from the C++ buffer.

    If np.asarray(buf) is truly zero-copy, then modifying the NumPy array
    should modify the underlying C++ buffer's memory. This is the
    'mutation visibility' proof of shared memory.
    """
    buf = fastreplay.RingBuffer(4)
    buf.push(100)
    buf.push(200)
    buf.push(300)

    # Get a zero-copy view
    arr = np.asarray(buf)

    # Mutate through the NumPy side
    arr[0] = 999

    # Read back through a fresh NumPy view from the C++ buffer
    arr2 = np.asarray(buf)
    assert arr2[0] == 999, (
        f"Mutation not visible: wrote 999 via NumPy view, "
        f"but C++ buffer still shows {arr2[0]}. "
        f"This means np.asarray(buf) is NOT zero-copy."
    )

    # Also verify via pop (reads from C++ internal memory)
    val = buf.pop()
    assert val == 999, (
        f"Mutation not visible via pop(): expected 999, got {val}. "
        f"The NumPy view does not share memory with the C++ buffer."
    )