"""
utils/logger.py — Centralised logging for the application.

All modules import `get_logger(__name__)` so every log line is tagged
with the originating module name.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured to write to stdout with a clear format."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger
