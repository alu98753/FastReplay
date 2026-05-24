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