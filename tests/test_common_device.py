"""Tests for common device utility."""

from src.common.device import resolve_device


def test_resolve_device_auto():
    """Verify resolve_device with auto returns a valid device string."""
    dev = resolve_device("auto")
    assert dev in ("cpu", "cuda", "mps")


def test_resolve_device_explicit():
    """Verify explicit device preference is respected or mapped accurately."""
    assert resolve_device("cpu") == "cpu"
