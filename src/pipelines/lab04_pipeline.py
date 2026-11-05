"""Backward-compatibility shim for Lab 04 pipeline.

Forwarded to: src.labs.lab04_wavelet.pipeline
"""

from src.labs.lab04_wavelet.pipeline import Lab04Pipeline

__all__ = ["Lab04Pipeline"]

if __name__ == "__main__":
    pipeline = Lab04Pipeline()
    pipeline.run_all()
