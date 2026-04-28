"""
TFLite-based object detector with built-in NMS.

Returns a list of Detection(box, score) sorted by descending score.
All coordinates are in pixel space relative to the *original* frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from anpr.config import DetectionConfig
from anpr.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Detection:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    score: float

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    def padded(self, pad: int, frame_w: int, frame_h: int) -> "Detection":
        return Detection(
            xmin=max(0, self.xmin - pad),
            ymin=max(0, self.ymin - pad),
            xmax=min(frame_w, self.xmax + pad),
            ymax=min(frame_h, self.ymax + pad),
            score=self.score,
        )

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.ymin : self.ymax, self.xmin : self.xmax]


def _compute_iou(box_a: Detection, box_b: Detection) -> float:
    ix1 = max(box_a.xmin, box_b.xmin)
    iy1 = max(box_a.ymin, box_b.ymin)
    ix2 = min(box_a.xmax, box_b.xmax)
    iy2 = min(box_a.ymax, box_b.ymax)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = box_a.width * box_a.height
    area_b = box_b.width * box_b.height
    return inter / (area_a + area_b - inter)


def _nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy NMS — keeps highest-score box, suppresses overlapping ones."""
    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for det in sorted_dets:
        if all(_compute_iou(det, k) < iou_threshold for k in kept):
            kept.append(det)
    return kept


class PlateDetector:
    def __init__(self, config: DetectionConfig) -> None:
        self._cfg = config
        log.info("Loading TFLite model: %s", config.model_path)
        self._interpreter = tf.lite.Interpreter(model_path=config.model_path)
        self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._input_idx = input_details[0]["index"]
        self._model_h: int = input_details[0]["shape"][1]
        self._model_w: int = input_details[0]["shape"][2]
        self._float_input: bool = input_details[0]["dtype"] == np.float32
        log.info(
            "Model input: %dx%d  float=%s",
            self._model_w,
            self._model_h,
            self._float_input,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on *frame* (BGR, HWC).
        Returns NMS-filtered Detection list sorted by score descending.
        """
        im_h, im_w = frame.shape[:2]

        # --- preprocess ---
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._model_w, self._model_h))
        tensor = np.expand_dims(resized, axis=0)
        if self._float_input:
            tensor = (np.float32(tensor) - 127.5) / 127.5

        # --- inference ---
        self._interpreter.set_tensor(self._input_idx, tensor)
        self._interpreter.invoke()

        outputs = [
            self._interpreter.get_tensor(o["index"])
            for o in self._output_details
        ]
        scores = outputs[0][0]
        boxes = outputs[1][0]  # [ymin, xmin, ymax, xmax] normalized

        # --- decode + threshold ---
        raw: list[Detection] = []
        for score, box in zip(scores, boxes):
            if score <= self._cfg.confidence_threshold:
                continue
            ymin = int(max(0, box[0] * im_h))
            xmin = int(max(0, box[1] * im_w))
            ymax = int(min(im_h, box[2] * im_h))
            xmax = int(min(im_w, box[3] * im_w))
            if xmax <= xmin or ymax <= ymin:
                continue
            raw.append(Detection(xmin, ymin, xmax, ymax, float(score)))

        return _nms(raw, self._cfg.nms_iou_threshold)
