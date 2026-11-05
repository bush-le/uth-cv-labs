"""Backward-compatibility shim for Lab 01 pipeline.

Forwarded to: src.labs.lab01_basics.pipeline
"""

from src.labs.lab01_basics.pipeline import run_lab01

__all__ = ["run_lab01"]

if __name__ == "__main__":
    run_lab01()
