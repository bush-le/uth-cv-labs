"""Centralized logging module for computer vision pipelines.

Provides standardized logging configurations without polluting stdout with raw print statements.
"""

from __future__ import annotations

import logging
import sys


def get_logger(
    name: str = "uth_cv",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Initialize or retrieve a structured logger instance.

    Args:
        name: Name of the logger, typically __name__ or module identifier.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to append log outputs.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
