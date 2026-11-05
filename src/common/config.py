"""Configuration utilities for loading and parsing YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid YAML dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data
