"""
Access control — loads authorized plates from file with hot-reload.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from anpr import plate as plate_mod
from anpr.config import AccessConfig
from anpr.utils.logger import get_logger

log = get_logger(__name__)

_STARTER_CONTENT = """\
# Authorized plates — one per line. Lines starting with # are ignored.
# Accepted formats: MH12AB1234, KA05MJ7777, etc.
"""

class AccessController:
    """
    Maintains the set of authorized plates and logs access events.
    """

    def __init__(self, config: AccessConfig) -> None:
        self._cfg = config
        self._plates: set[str] = set()
        self._plates_lock = threading.RLock()
        self._last_mtime: float = 0.0

        self._ensure_plates_file()
        self.reload_plates()  # Initial load

        self._stop_event = threading.Event()
        self._watcher = threading.Thread(
            target=self._watch_loop, daemon=True, name="plates-watcher"
        )
        self._watcher.start()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def is_authorized(self, plate: str) -> bool:
        with self._plates_lock:
            return plate in self._plates

    def get_status(self, plate: str) -> str:
        return "AUTHORIZED" if self.is_authorized(plate) else "UNAUTHORIZED"

    def plate_count(self) -> int:
        with self._plates_lock:
            return len(self._plates)

    def reload_plates(self) -> None:
        """Exposed method for manual or automatic reloads."""
        path = Path(self._cfg.authorized_plates_file)
        try:
            mtime = path.stat().st_mtime
            # If triggered manually, we ignore the mtime check
            content = path.read_text(encoding="utf-8")
            new_plates = plate_mod.load_plates_from_text(content)
            
            with self._plates_lock:
                self._plates = new_plates
                self._last_mtime = mtime
            log.info("Reloaded %d authorized plate(s).", len(new_plates))
        except Exception as exc:
            log.error("Failed to reload %s: %s", path, exc)

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _ensure_plates_file(self) -> None:
        path = Path(self._cfg.authorized_plates_file)
        if not path.exists():
            path.write_text(_STARTER_CONTENT, encoding="utf-8")
            log.info("Created starter authorized_plates file: %s", path)

    def _watch_loop(self) -> None:
        while not self._stop_event.wait(self._cfg.reload_interval_seconds):
            # Check file mtime for automatic hot-reload
            path = Path(self._cfg.authorized_plates_file)
            if path.exists() and path.stat().st_mtime != self._last_mtime:
                self.reload_plates()