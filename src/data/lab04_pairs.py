"""Prepare source-disjoint Lab04 pairs without changing source images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageEnhance

from src.common.logger import get_logger

logger = get_logger(__name__)


def load_pair_dataset(root: Path, dataset_dir: Path) -> tuple:
    """Read prepared manifests and RGB/gray images without regenerating data.

    Returns:
        Sources, metadata, split pairs, RGB uint8 images and grayscale comparison images.
    """

    def read_csv(name: str) -> list[dict[str, str]]:
        with (dataset_dir / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    sources = read_csv("sources.csv")
    metadata = {row["image_id"]: row for row in read_csv("images.csv")}
    pairs = {
        split: read_csv(f"pairs_{split}.csv") for split in ("calibration", "evaluation")
    }
    rgb_images, gray_images = {}, {}
    for image_id, row in metadata.items():
        with Image.open(root / row["path"]) as image:
            rgb_images[image_id] = np.array(image.convert("RGB"), dtype=np.uint8)
            gray_images[image_id] = np.array(image.convert("L"), dtype=np.uint8)
    return sources, metadata, pairs, rgb_images, gray_images


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Write nonempty metadata records to a UTF-8 CSV file."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def prepare_pairs(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Create RGB uint8 (H,W,3) variants and source-disjoint pair manifests.

    Args:
        config: Source categories, paths, seed and transformation parameters.
        root: Project root used to resolve relative dataset paths.

    Returns:
        Counts of sources, generated images and labels per split.
        Source files are read only; outputs are reproducible for a fixed input.
    """
    source_root = root / config["source_root"]
    output = root / config["output_root"]
    rng = np.random.default_rng(config["seed"])
    sources = []
    digests: set[str] = set()
    expected: set[Path] = set()
    for theme, categories in config["categories"].items():
        for category, numbers in categories.items():
            files = [source_root / theme / f"doi_tuong_{n:02d}.png" for n in numbers]
            count = config["calibration_per_category"]
            if not 0 < count < len(files):
                raise ValueError(
                    "Each category needs calibration and evaluation sources."
                )
            calibration = set(rng.permutation(len(files))[:count].tolist())
            for index, path in enumerate(files):
                if path in expected:
                    raise ValueError(f"Source assigned more than once: {path}")
                expected.add(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in digests:
                    raise ValueError(f"Duplicate source content: {path}")
                digests.add(digest)
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                sources.append(
                    {
                        "source_id": f"{theme}_{path.stem}",
                        "theme": theme,
                        "category": category,
                        "split": (
                            "calibration" if index in calibration else "evaluation"
                        ),
                        "path": path.relative_to(root).as_posix(),
                        "sha256": digest,
                        "width": width,
                        "height": height,
                    }
                )
    actual = {
        p for theme in config["categories"] for p in (source_root / theme).glob("*.png")
    }
    if actual != expected:
        raise ValueError(
            "Source files differ from category mapping; review config first."
        )
    if not 0 <= config["noise_std"] or not 1 <= config["jpeg_quality"] <= 95:
        raise ValueError("Invalid noise standard deviation or JPEG quality.")
    output.mkdir(parents=True, exist_ok=True)
    images = []
    pairs = []
    for source in sources:
        with Image.open(root / source["path"]) as raw:
            original = raw.convert("RGB")
        images.append(
            {**source, "image_id": source["source_id"], "transform": "original"}
        )
        variants = {
            name: ImageEnhance.Brightness(original).enhance(factor)
            for name, factor in config["brightness_factors"].items()
        }
        pixels = np.asarray(original, dtype=np.float32)
        noisy = np.clip(
            pixels + rng.normal(0, config["noise_std"], pixels.shape), 0, 255
        )
        variants["noise"] = Image.fromarray(noisy.astype(np.uint8))
        variants["jpeg"] = original
        for name, image in variants.items():
            image_id = f"{source['source_id']}__{name}"
            path = (
                output
                / source["split"]
                / source["theme"]
                / f"{image_id}.{'jpg' if name == 'jpeg' else 'png'}"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(
                path, **({"quality": config["jpeg_quality"]} if name == "jpeg" else {})
            )
            images.append(
                {
                    **source,
                    "path": path.relative_to(root).as_posix(),
                    "image_id": image_id,
                    "transform": name,
                }
            )
            pairs.append(
                {
                    "split": source["split"],
                    "theme": source["theme"],
                    "left_id": source["source_id"],
                    "right_id": image_id,
                    "left_path": source["path"],
                    "right_path": path.relative_to(root).as_posix(),
                    "label": 1,
                    "pair_type": "same_source_variant",
                }
            )
    for left, right in combinations(sources, 2):
        if (
            left["split"] == right["split"]
            and left["theme"] == right["theme"]
            and left["category"] != right["category"]
        ):
            pairs.append(
                {
                    "split": left["split"],
                    "theme": left["theme"],
                    "left_id": left["source_id"],
                    "right_id": right["source_id"],
                    "left_path": left["path"],
                    "right_path": right["path"],
                    "label": 0,
                    "pair_type": "different_category_same_theme",
                }
            )
    summary = {
        "seed": config["seed"],
        "sources": len(sources),
        "images": len(images),
        "variants": len(images) - len(sources),
        "splits": {},
    }
    for split in ["calibration", "evaluation"]:
        selected = [pair for pair in pairs if pair["split"] == split]
        summary["splits"][split] = {
            "sources": sum(s["split"] == split for s in sources),
            "positive_pairs": sum(p["label"] == 1 for p in selected),
            "negative_pairs": sum(p["label"] == 0 for p in selected),
        }
        write_records(output / f"pairs_{split}.csv", selected)
    write_records(output / "sources.csv", sources)
    write_records(output / "images.csv", images)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "generation_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    """Run dataset preparation from the project root or any working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/dataset/lab04_pairs.yaml")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    path = args.config if args.config.is_absolute() else root / args.config
    summary = prepare_pairs(yaml.safe_load(path.read_text(encoding="utf-8")), root)
    logger.info("Lab04 dataset prepared: %s", summary)


if __name__ == "__main__":
    main()
