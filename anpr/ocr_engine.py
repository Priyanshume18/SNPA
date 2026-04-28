"""
OCR engine for license plates.

Pipeline
--------
1. Convert crop to grayscale
2. Adaptive preprocessing stack (upscale → sharpen → denoise → equalize → threshold)
3. EasyOCR read with alphanumeric allowlist
4. Candidate selection by regex extraction + confidence
5. Temporal majority-vote smoothing across recent frames per location bucket
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

import cv2
import easyocr
import numpy as np
import torch  # ✅ NEW

from anpr import plate as plate_mod
from anpr.config import OCRConfig
from anpr.utils.logger import get_logger

log = get_logger(__name__)

# Allowlist for EasyOCR — Indian plates only use uppercase alpha + digits
_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _preprocess(crop: np.ndarray, upscale: float) -> np.ndarray:
    """Return a cleaned grayscale image optimized for plate OCR."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if upscale != 1.0:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_morph)

    return thresh


class PlateOCR:
    def __init__(self, config: OCRConfig) -> None:
        self._cfg = config

        # ---------------- GPU DETECTION ----------------
        use_gpu = False

        if config.use_gpu:
            try:
                if torch.cuda.is_available():
                    use_gpu = True
                    log.info("CUDA available — using GPU: %s", torch.cuda.get_device_name(0))
                else:
                    log.warning("GPU requested but CUDA not available. Falling back to CPU.")
            except Exception as e:
                log.warning("Error checking CUDA: %s. Falling back to CPU.", e)

        log.info("Initialising EasyOCR (gpu=%s)…", use_gpu)

        self._reader = easyocr.Reader(
            config.languages,
            gpu=use_gpu
        )

        # ---------------- GPU WARM-UP ----------------
        if use_gpu:
            try:
                dummy = np.zeros((64, 256), dtype=np.uint8)
                self._reader.readtext(dummy, allowlist=_ALLOWLIST)
                log.info("GPU warm-up completed")
            except Exception as e:
                log.warning("GPU warm-up failed: %s", e)

        # plate_key → deque of recent plate strings
        self._history: dict[tuple, deque] = defaultdict(
            lambda: deque(maxlen=config.history_size)
        )

    def read(
        self,
        crop: np.ndarray,
        location_bucket: tuple[int, int],
    ) -> tuple[Optional[str], float]:

        try:
            preprocessed = _preprocess(crop, self._cfg.upscale_factor)
        except Exception as exc:
            log.warning("Preprocessing failed: %s", exc)
            return None, 0.0

        try:
            results = self._reader.readtext(
                preprocessed,
                allowlist=_ALLOWLIST,
                min_size=10,
                text_threshold=self._cfg.min_confidence,
            )
        except Exception as exc:
            log.warning("EasyOCR error: %s", exc)
            return None, 0.0

        # --- pick best candidate ---
        best_text = ""
        best_score = 0.0

        for (_, text, prob) in results:
            candidate = plate_mod.extract_plate(text)
            if candidate and prob > best_score:
                best_text = candidate
                best_score = float(prob)

        # --- fallback ---
        if not best_text and results:
            combined = "".join(r[1] for r in results)
            best_text = plate_mod.extract_plate(combined)
            if best_text:
                best_score = min(r[2] for r in results)

        # --- temporal smoothing ---
        if best_text:
            self._history[location_bucket].append(best_text)

        history = self._history[location_bucket]

        if len(history) >= self._cfg.min_votes:
            final = max(set(history), key=history.count)
        else:
            final = best_text or None

        if final and not plate_mod.is_valid(final):
            final = None

        return final, best_score

    def reset_bucket(self, location_bucket: tuple[int, int]) -> None:
        self._history.pop(location_bucket, None)