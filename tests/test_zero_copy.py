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
    view_arr = buf.pop_view()

    assert view_arr.shape == (3,)
    assert view_arr.strides == base_arr.strides
    assert view_arr[0] == 0
    assert view_arr[2] == 2

    assert np.share_memory(base_arr, view_arr)

def test_zero_copy_pop_view_wrap_around():
    buf = fastreplay.RingBuffer(4)
    for i in range(4):
        buf.push(i*10)
    buf.pop()
    buf.push(50)
    buf.push(60)

    # wrap around
    base_arr = np.asarray(buf)
    view_arr = buf.pop_view(4)

    assert view_arr.tolist() == [20, 30, 40, 50]
    assert np.shares_memory(base_arr, view_arr) is False

def test_pop_view_insufficient_data():
    buf = fastreplay.RingBuffer(4)
    buf.push(10)
    buf.push(20)

    with pytest.raises(RuntimeError, match="Not enough data"):
        buf.pop_view(3)


#TODO : Dangling pointer
    