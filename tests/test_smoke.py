"""Week 2 tests: 驗證 fastreplay 模組與 RingBuffer 功能。"""
import pytest
import fastreplay

def test_import():
    """Import fastreplay module."""
    assert hasattr(fastreplay, "RingBuffer")
    assert hasattr(fastreplay, "add")

def test_add():
    """Smoke test: C++ add() 透過 pybind11 正確回傳。"""
    assert fastreplay.add(2, 3) == 5

# --- week2 Ring Buffer test ---

def test_capacity():
    rb = fastreplay.RingBuffer(2)
    assert rb.capacity() == 2

def test_empty():
    rb = fastreplay.RingBuffer(0)
    assert rb.empty() == True

def test_size():
    rb = fastreplay.RingBuffer(4)
    assert not rb.size()

def test_ringbuffer_create():
    """
    RingBuffer constructor test.
    Test the invariant when rb is constructed
    """
    rb = fastreplay.RingBuffer(4)
    assert rb.capacity() == 4
    assert rb.size() == 0
    assert rb.empty()

def test_ringbuffer_push_pop():
    """RingBuffer push and pop test."""
    rb = fastreplay.RingBuffer(4)
    assert rb.push(1) is True
    assert rb.size() == 1
    
    val = rb.pop()
    assert val == 1
    assert rb.size() == 0
    assert rb.empty()

def test_ringbuffer_fifo_order():
    """RingBuffer FIFO order test."""
    rb = fastreplay.RingBuffer(4)
    rb.push(10)
    rb.push(20)
    rb.push(30)
    assert rb.pop() == 10
    assert rb.pop() == 20
    assert rb.pop() == 30
    assert rb.empty()

def test_ringbuffer_pop_from_empty():
    """RingBuffer pop from empty test."""
    rb = fastreplay.RingBuffer(4)
    assert rb.pop() is None
    assert rb.size() == 0
    assert rb.empty()

def test_ringbuffer_push_to_full():
    """RingBuffer push to full test."""
    rb = fastreplay.RingBuffer(3)
    assert rb.push(1)
    assert rb.push(2)
    assert rb.push(3)
    assert not rb.push(4)
    assert rb.size() == 3

def test_ringbuffer_wrap_around():
    """RingBuffer wrap around test."""
    rb = fastreplay.RingBuffer(3)
    rb.push(1)
    rb.push(2)
    rb.push(3)
    assert rb.pop() == 1
    assert rb.pop() == 2
    rb.push(4)
    rb.push(5)
    assert rb.pop() == 3
    assert rb.pop() == 4
    assert rb.pop() == 5
    assert rb.empty()

def test_ringbuffer_size_tracks_correctly():
    """RingBuffer size tracking test."""
    rb = fastreplay.RingBuffer(4)
    assert rb.size() == 0
    rb.push(1)
    assert rb.size() == 1
    rb.push(2)
    assert rb.size() == 2
    rb.pop()
    assert rb.size() == 1
    rb.pop()
    assert rb.size() == 0
