"""
Access logger — writes structured CSV rows with deduplication.

Deduplication: the same plate is not logged again within DEDUP_SECONDS,
preventing log floods when a stationary vehicle stays in frame.
"""

from __future__ import annotations

import csv
import threading
from datetime import datetime
from pathlib import Path

from anpr.config import AccessConfig, LoggingConfig
from anpr.utils.logger import get_logger

log = get_logger(__name__)

_CSV_HEADERS = ["timestamp", "plate", "status", "ocr_confidence", "frame_no"]


class AccessLogger:
    def __init__(self, access_cfg: AccessConfig, log_cfg: LoggingConfig) -> None:
        self._path = Path(log_cfg.log_file)
        self._dedup_seconds = access_cfg.dedup_seconds
        self._last_seen: dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._ensure_file()

    # ------------------------------------------------------------------ #

    def maybe_log(
        self,
        plate: str,
        status: str,
        confidence: float,
        frame_no: int,
    ) -> bool:
        """
        Log plate access event if outside dedup window.
        Returns True if the event was actually written.
        """
        with self._lock:
            if not self._should_log(plate):
                return False
            self._write(plate, status, confidence, frame_no)
            return True

    # ------------------------------------------------------------------ #

    def _should_log(self, plate: str) -> bool:
        now = datetime.now()
        prev = self._last_seen.get(plate)
        if prev is None or (now - prev).total_seconds() >= self._dedup_seconds:
            self._last_seen[plate] = now
            return True
        return False

    def _write(
        self,
        plate: str,
        status: str,
        confidence: float,
        frame_no: int,
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self._path, "a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(
                    [timestamp, plate, status, f"{confidence:.4f}", frame_no]
                )
            log.info("[ACCESS] %s | %s | conf=%.2f", plate, status, confidence)
        except OSError as exc:
            log.error("Failed to write access log: %s", exc)

    def _ensure_file(self) -> None:
        if self._path.exists():
            return
        try:
            with open(self._path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(_CSV_HEADERS)
            log.info("Created access log: %s", self._path)
        except OSError as exc:
            log.error("Cannot create log file %s: %s", self._path, exc)
