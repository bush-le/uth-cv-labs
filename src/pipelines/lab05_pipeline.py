"""Backward-compatibility shim for Lab 05 pipeline.

Forwarded to: src.labs.lab05_facenet.pipeline
"""

from src.labs.lab05_facenet.pipeline import run_lab05, run_lab05_pipeline

__all__ = ["run_lab05", "run_lab05_pipeline"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lab 05 Master Pipeline (Legacy Shim)")
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--config", default="configs/experiment/lab05_facenet.yaml")
    parser.add_argument("--output-dir", default="experiments/predictions/lab05")
    args = parser.parse_args()

    run_lab05_pipeline(config_path=args.config, mode=args.mode, output_dir=args.output_dir)
