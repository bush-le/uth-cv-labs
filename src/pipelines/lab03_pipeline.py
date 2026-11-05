"""Backward-compatibility shim for Lab 03 pipeline.

Forwarded to: src.labs.lab03_canny.pipeline
"""

from src.labs.lab03_canny.pipeline import Lab03Pipeline

__all__ = ["Lab03Pipeline"]

if __name__ == "__main__":
    pipeline = Lab03Pipeline()
    pipeline.run_all()
