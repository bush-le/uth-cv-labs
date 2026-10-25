"""Verify pair labels, source separation and reproducible preparation."""

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.lab04_pairs import prepare_pairs


@pytest.fixture
def dataset_config(tmp_path: Path) -> dict:
    """Create eight distinct RGB uint8 (12,12,3) source images."""
    folder = tmp_path / "samples/topic"
    folder.mkdir(parents=True)
    for number in range(1, 9):
        pixels = np.random.default_rng(number).integers(
            0, 256, (12, 12, 3), dtype=np.uint8
        )
        Image.fromarray(pixels).save(folder / f"doi_tuong_{number:02d}.png")
    return {
        "source_root": "samples",
        "output_root": "processed",
        "seed": 42,
        "calibration_per_category": 2,
        "brightness_factors": {"bright": 1.3, "dark": 0.7},
        "noise_std": 10,
        "jpeg_quality": 50,
        "categories": {"topic": {"animal": [1, 2, 3, 4], "food": [5, 6, 7, 8]}},
    }


@pytest.mark.smoke
def test_prepare_pairs_smoke(tmp_path: Path, dataset_config: dict) -> None:
    """Prepare the expected counts without changing sources."""
    summary = prepare_pairs(dataset_config, tmp_path)
    assert summary["sources"] == 8
    assert summary["variants"] == 32
    assert summary["splits"]["evaluation"]["negative_pairs"] == 4


def test_labels_and_split_isolation(tmp_path: Path, dataset_config: dict) -> None:
    """All pairs stay within one source split and obey category-based labels."""
    prepare_pairs(dataset_config, tmp_path)
    with (tmp_path / "processed/images.csv").open(encoding="utf-8") as stream:
        images = {row["image_id"]: row for row in csv.DictReader(stream)}
    memberships = {}
    for split in ["calibration", "evaluation"]:
        with (tmp_path / f"processed/pairs_{split}.csv").open(
            encoding="utf-8"
        ) as stream:
            pairs = list(csv.DictReader(stream))
        for pair in pairs:
            left, right = images[pair["left_id"]], images[pair["right_id"]]
            assert left["split"] == right["split"] == split
            if pair["label"] == "1":
                assert left["source_id"] == right["source_id"]
            else:
                assert left["source_id"] != right["source_id"]
                assert left["category"] != right["category"]
        memberships[split] = {
            row["source_id"] for row in images.values() if row["split"] == split
        }
    assert memberships["calibration"].isdisjoint(memberships["evaluation"])


def test_reproducible_and_sources_unchanged(
    tmp_path: Path, dataset_config: dict
) -> None:
    """A repeat run reproduces every artifact while preserving original bytes."""

    def hashes(folder: Path) -> dict[str, str]:
        return {
            p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*")
            if p.is_file()
        }

    originals = hashes(tmp_path / "samples")
    prepare_pairs(dataset_config, tmp_path)
    first = hashes(tmp_path / "processed")
    prepare_pairs(dataset_config, tmp_path)
    assert first == hashes(tmp_path / "processed")
    assert originals == hashes(tmp_path / "samples")
