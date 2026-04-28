"""
ANPR Pipeline — orchestrates camera, motion gate, detector, OCR, access control, logging.

Threading model
---------------
  Main thread  : capture → motion gate (bool) → detector → enqueue crops
  OCR thread   : dequeue crops → OCR → enqueue results
  Main thread  : dequeue results → draw overlay → display

Stale result expiry
-------------------
  Every frame the detector reports which screen buckets have an active plate.
  Any bucket absent for RESULT_TTL_FRAMES consecutive frames is evicted from
  both the overlay cache AND the OCR engine's temporal-smoothing history.
  This prevents ghost labels from old plates persisting on screen.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from anpr.motion_gate import PipelineGate
from anpr.access_control import AccessController
from anpr.access_logger import AccessLogger
from anpr.config import Config
from anpr.detector import Detection, PlateDetector
from anpr.ocr_engine import PlateOCR
from anpr.utils.logger import get_logger

log = get_logger(__name__)

_STOP = object()
_STATS_INTERVAL_FRAMES = 300

# Frames a bucket must be absent before its result and OCR history are wiped.
# At 30 fps → 10 frames ≈ 0.33 s.
RESULT_TTL_FRAMES = 3


@dataclass
class _CropTask:
    frame: np.ndarray
    detection: Detection
    frame_no: int
    location_bucket: tuple[int, int]


@dataclass
class _OCRResult:
    detection: Detection
    plate: Optional[str]
    confidence: float
    status: str
    frame_no: int


class ANPRPipeline:
    def __init__(self, config: Config) -> None:
        self._cfg = config

        # PipelineGate.check(frame) → bool
        self._gate = PipelineGate(config.gate)
        self._detector = PlateDetector(config.detection)
        self._ocr = PlateOCR(config.ocr)
        self._access = AccessController(config.access)
        self._logger = AccessLogger(config.access, config.logging)

        self._crop_queue: queue.Queue[object] = queue.Queue(maxsize=8)
        self._result_queue: queue.Queue[_OCRResult] = queue.Queue(maxsize=32)

        self._running = False
        self._frame_no = 0

        # bucket → most recent OCR result
        self._latest_results: dict[tuple, _OCRResult] = {}
        # bucket → last frame_no the detector saw a plate there
        self._bucket_last_seen: dict[tuple, int] = {}

    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self._running = True

        ocr_thread = threading.Thread(
            target=self._ocr_worker, daemon=True, name="ocr-worker"
        )
        ocr_thread.start()
        log.info("OCR worker thread started")

        cap = self._open_camera()
        if cap is None:
            self.stop()
            return

        log.info(
            "Camera opened. Authorized plates: %d. Press 'q' to quit.",
            self._access.plate_count(),
        )

        fps_limit = self._cfg.camera.fps_limit
        frame_interval = (1.0 / fps_limit) if fps_limit else 0.0

        try:
            self._capture_loop(cap, frame_interval)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._crop_queue.put(_STOP)
            ocr_thread.join(timeout=5)
            self._access.stop()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ #

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        source = self._cfg.camera_source()
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            log.error("Cannot open camera/source: %s", source)
            return None
        return cap

    def _capture_loop(self, cap: cv2.VideoCapture, frame_interval: float) -> None:
        ocr_interval = self._cfg.ocr.interval_frames
        cfg_det = self._cfg.detection

        while self._running:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame read failed — end of stream or camera error")
                break

            self._frame_no += 1
            self._drain_results()

            # ── Motion gate (returns bool) ────────────────────────────────
            gate_active = self._gate.check(frame)

            detections = []
            if gate_active:
                detections = self._detector.detect(frame)

            run_ocr = self._frame_no % ocr_interval == 0

            for det in detections:
                padded = det.padded(cfg_det.pad_pixels, frame.shape[1], frame.shape[0])
                if (
                    padded.width < cfg_det.min_plate_width_px
                    or padded.height < cfg_det.min_plate_height_px
                ):
                    continue

                bucket = (det.xmin // 50, det.ymin // 50)
                self._bucket_last_seen[bucket] = self._frame_no

                if run_ocr:
                    crop = padded.crop(frame)
                    try:
                        self._crop_queue.put_nowait(
                            _CropTask(crop, det, self._frame_no, bucket)
                        )
                    except queue.Full:
                        pass

                result = self._latest_results.get(bucket)
                self._draw_overlay(frame, det, result)

            # ── Expire stale buckets ──────────────────────────────────────
            # Buckets not seen by the detector for RESULT_TTL_FRAMES frames
            # are evicted: overlay label removed AND OCR history cleared.
            # Without the OCR history clear, old plate votes would pollute
            # the majority-vote for the next plate in the same screen region.
            stale = [
                b for b, last in self._bucket_last_seen.items()
                if (self._frame_no - last) > RESULT_TTL_FRAMES
            ]
            for b in stale:
                self._latest_results.pop(b, None)
                self._bucket_last_seen.pop(b, None)
                self._ocr.reset_bucket(b)
                log.debug("Evicted stale bucket %s", b)

            self._draw_stats(frame, len(detections), gate_active)

            if self._cfg.camera.display:
                cv2.imshow(self._cfg.camera.display_window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("User requested exit")
                    break

            elapsed = time.monotonic() - t0
            if frame_interval > elapsed:
                time.sleep(frame_interval - elapsed)

    # ------------------------------------------------------------------ #

    def _ocr_worker(self) -> None:
        while True:
            task = self._crop_queue.get()
            if task is _STOP:
                log.info("OCR worker stopping")
                break
            if not isinstance(task, _CropTask):
                continue

            plate_str, confidence = self._ocr.read(task.frame, task.location_bucket)

            status = "UNKNOWN"
            if plate_str:
                status = self._access.get_status(plate_str)
                self._logger.maybe_log(plate_str, status, confidence, task.frame_no)

            try:
                self._result_queue.put_nowait(
                    _OCRResult(task.detection, plate_str, confidence, status, task.frame_no)
                )
            except queue.Full:
                pass

    def _drain_results(self) -> None:
        while True:
            try:
                result = self._result_queue.get_nowait()
                det = result.detection
                bucket = (det.xmin // 50, det.ymin // 50)
                self._latest_results[bucket] = result
            except queue.Empty:
                break

    # ------------------------------------------------------------------ #

    _STATUS_COLORS = {
        "AUTHORIZED":   (0, 220, 0),
        "UNAUTHORIZED": (0, 0, 230),
        "UNKNOWN":      (180, 180, 0),
        "INVALID":      (128, 128, 128),
    }

    def _draw_overlay(
        self,
        frame: np.ndarray,
        det: Detection,
        result: Optional[_OCRResult],
    ) -> None:
        status = result.status if result else "UNKNOWN"
        color = self._STATUS_COLORS.get(status, (200, 200, 200))
        cv2.rectangle(frame, (det.xmin, det.ymin), (det.xmax, det.ymax), color, 2)

        if result and result.plate:
            label = f"{result.plate}  [{status}]  {result.confidence:.0%}"
            text_y = max(20, det.ymin - 10)
            cv2.putText(frame, label, (det.xmin + 1, text_y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, label, (det.xmin, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"{det.score:.0%}",
                    (det.xmin, det.ymax + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def _draw_stats(self, frame: np.ndarray, n_detections: int, gate_active: bool) -> None:
        h = frame.shape[0]
        status_text = "ACTIVE" if gate_active else "IDLE (Gate Closed)"
        status_color = (0, 255, 0) if gate_active else (100, 100, 100)
        info = (
            f"Frame #{self._frame_no}  |  "
            f"Status: {status_text}  |  "
            f"Detections: {n_detections}  |  "
            f"Auth: {self._access.plate_count()}"
        )
        cv2.putText(frame, info, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)