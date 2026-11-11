"""Hardware device resolution utility for PyTorch workflows."""

from __future__ import annotations

import torch


def resolve_device(device_pref: str = "auto") -> str:
    """Resolve compute device based on preference and hardware availability.

    Args:
        device_pref: Preferred device string ('auto', 'cuda', 'mps', 'cpu').

    Returns:
        Resolved device string ('cuda', 'mps', or 'cpu').
    """
    pref = (device_pref or "auto").strip().lower()
    if pref in ("cuda", "gpu"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        return "mps" if mps_available else "cpu"
    if pref == "cpu":
        return "cpu"

    # auto mode
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
