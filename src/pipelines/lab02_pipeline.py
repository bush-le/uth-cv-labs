"""Backward-compatibility shim for Lab 02 pipeline.

Forwarded to: src.labs.lab02_filters.pipeline
"""

from src.labs.lab02_filters.pipeline import run_lab02

__all__ = ["run_lab02"]

if __name__ == "__main__":
    run_lab02()
