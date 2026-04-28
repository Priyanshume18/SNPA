"""
Centralised logger factory.
All modules call get_logger(__name__) — output goes to both console and file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_LOG_FILE = "anpr.log"
_LEVEL = logging.INFO


def configure(level: str = "INFO", log_file: str = "anpr.log") -> None:
    global _CONFIGURED, _LOG_FILE, _LEVEL
    _LEVEL = getattr(logging, level.upper(), logging.INFO)
    _LOG_FILE = log_file
    _CONFIGURED = True

    root = logging.getLogger()
    if root.handlers:
        return  # already set up

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass  # non-fatal if log file can't be opened

    root.setLevel(_LEVEL)


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure()
    return logging.getLogger(name)
