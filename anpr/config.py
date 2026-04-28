"""
Configuration — single source of truth for all tunables.
Loads from config.yaml; falls back to sane defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclass
class DetectionConfig:
    model_path: str = "detect.tflite"
    confidence_threshold: float = 0.50
    nms_iou_threshold: float = 0.45
    min_plate_width_px: int = 120
    min_plate_height_px: int = 40
    pad_pixels: int = 10


@dataclass
class OCRConfig:
    interval_frames: int = 5
    languages: list = field(default_factory=lambda: ["en"])
    use_gpu: bool = True
    upscale_factor: float = 2.0
    history_size: int = 7
    min_votes: int = 3
    min_confidence: float = 0.35


@dataclass
class AccessConfig:
    authorized_plates_file: str = "authorized_plates.txt"
    reload_interval_seconds: int = 30
    dedup_seconds: int = 20


@dataclass
class LoggingConfig:
    log_file: str = "access_logs.csv"
    app_log_file: str = "anpr.log"
    level: str = "INFO"


@dataclass
class CameraConfig:
    source: str = "0"
    display: bool = True
    display_window_name: str = "ANPR"
    fps_limit: Optional[int] = None

# --- NEW SECTION START ---
@dataclass
class GateConfig:
    enabled: bool = True
    motion_threshold: int = 600
    texture_threshold: float = 0.03
    resize_factor: float = 0.5
# --- NEW SECTION END ---


@dataclass
class Config:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    access: AccessConfig = field(default_factory=AccessConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    gate: GateConfig = field(default_factory=GateConfig) # Added attribute

    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str = "config.yaml") -> "Config":
        if not _YAML_AVAILABLE or not Path(path).exists():
            return cls()

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        def _merge(dc_cls, section_key):
            section = raw.get(section_key, {})
            obj = dc_cls()
            for k, v in section.items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            return obj

        return cls(
            detection=_merge(DetectionConfig, "detection"),
            ocr=_merge(OCRConfig, "ocr"),
            access=_merge(AccessConfig, "access"),
            logging=_merge(LoggingConfig, "logging"),
            camera=_merge(CameraConfig, "camera"),
            gate=_merge(GateConfig, "gate"), # Added to merge logic
        )

    def camera_source(self):
        """Return int camera index or str path/URL."""
        try:
            return int(self.camera.source)
        except ValueError:
            return self.camera.source