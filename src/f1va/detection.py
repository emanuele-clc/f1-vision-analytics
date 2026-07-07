"""Detection delle monoposto con Ultralytics YOLO.

Wrapper sottile attorno a YOLO per restituire detection in un formato neutro,
così il resto della pipeline non dipende dall'API di Ultralytics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    xyxy: np.ndarray      # (N, 4) bounding box in pixel
    confidence: np.ndarray  # (N,)
    class_id: np.ndarray    # (N,)


class CarDetector:
    def __init__(self, model: str = "yolo11n.pt", conf: float = 0.25,
                 iou: float = 0.5, classes=None, device: str = "auto") -> None:
        from ultralytics import YOLO  # import ritardato: pesante

        self.model = YOLO(model)
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.device = None if device == "auto" else device

    def detect(self, frame: np.ndarray) -> Detection:
        """Esegue il detector su un singolo frame BGR (OpenCV)."""
        res = self.model.predict(
            frame, conf=self.conf, iou=self.iou,
            classes=self.classes, device=self.device, verbose=False,
        )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            empty = np.empty((0,))
            return Detection(np.empty((0, 4)), empty, empty)
        return Detection(
            xyxy=boxes.xyxy.cpu().numpy(),
            confidence=boxes.conf.cpu().numpy(),
            class_id=boxes.cls.cpu().numpy().astype(int),
        )
