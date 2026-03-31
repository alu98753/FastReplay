"""Week 1 smoke test: 確認 pybind11 module 能被 import。"""
import pytest

def test_import():
    """Import fastreplay module."""
    import fastreplay
    assert hasattr(fastreplay, "add")

def test_add():
    """Smoke test: C++ add() 透過 pybind11 正確回傳。"""
    import fastreplay
    assert fastreplay.add(2, 3) == 5
    