"""Multi-object tracking con ByteTrack (via supervision)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import Detection


@dataclass
class Tracks:
    xyxy: np.ndarray        # (N, 4)
    tracker_id: np.ndarray  # (N,)
    class_id: np.ndarray    # (N,)


class Tracker:
    def __init__(self, track_thresh: float = 0.25, match_thresh: float = 0.8,
                 track_buffer: int = 30) -> None:
        import supervision as sv

        self._sv = sv
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            minimum_matching_threshold=match_thresh,
            lost_track_buffer=track_buffer,
        )

    def update(self, det: Detection) -> Tracks:
        sv = self._sv
        detections = sv.Detections(
            xyxy=det.xyxy, confidence=det.confidence, class_id=det.class_id,
        )
        tracked = self.tracker.update_with_detections(detections)
        return Tracks(
            xyxy=tracked.xyxy,
            tracker_id=tracked.tracker_id,
            class_id=tracked.class_id,
        )


def bottom_center(xyxy: np.ndarray) -> np.ndarray:
    """Punto di contatto a terra (base del box) per la proiezione omografica."""
    if xyxy.size == 0:
        return np.empty((0, 2))
    x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    y = xyxy[:, 3]
    return np.stack([x, y], axis=1)
