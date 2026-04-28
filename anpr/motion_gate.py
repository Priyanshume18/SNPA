import cv2
import numpy as np

class PipelineGate:
    def __init__(self, config):
        # We access config.gate.motion_threshold, etc.
        self._cfg = config
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

    def check(self, frame) -> bool:
        if not self._cfg.enabled:
            return True

        # 1. Motion Gate (Stage 1)
        scale = self._cfg.resize_factor
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        fgmask = self.fgbg.apply(small_frame)
        
        # Remove shadows (gray pixels)
        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        motion_count = np.sum(fgmask > 0)

        if motion_count < self._cfg.motion_threshold:
            return False

        # 2. Texture Pre-screener (Stage 2)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        motion_roi = cv2.bitwise_and(gray, gray, mask=fgmask)
        
        edges = cv2.Canny(motion_roi, 100, 200)
        edge_density = np.sum(edges > 0) / (small_frame.shape[0] * small_frame.shape[1])

        return edge_density >= self._cfg.texture_threshold