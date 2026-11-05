"""Backward-compatibility shim for Lab04 dataset pairs."""

from src.labs.lab04_wavelet.wavelet_pairs import *  # noqa: F401, F403

if __name__ == "__main__":
    from src.labs.lab04_wavelet.wavelet_pairs import main
    main()
