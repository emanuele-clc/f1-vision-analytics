"""Overlay sui frame e mini-mappa bird's-eye."""
from __future__ import annotations

import cv2
import numpy as np

from .tracking import Tracks


def draw_tracks(frame: np.ndarray, tracks: Tracks, speeds: dict[int, float]) -> np.ndarray:
    out = frame.copy()
    for box, tid in zip(tracks.xyxy, tracks.tracker_id):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
        label = f"#{int(tid)}"
        if int(tid) in speeds:
            label += f"  {speeds[int(tid)]:.0f} km/h"
        cv2.putText(out, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    return out


def draw_minimap(world_xy: np.ndarray, tracker_ids: np.ndarray,
                 size: int = 240, margin: float = 5.0) -> np.ndarray:
    """Vista dall'alto delle auto proiettate sulla mappa del circuito."""
    canvas = np.full((size, size, 3), 30, dtype=np.uint8)
    if world_xy.size == 0:
        return canvas
    mn = world_xy.min(axis=0) - margin
    mx = world_xy.max(axis=0) + margin
    span = np.maximum(mx - mn, 1e-6)
    for xy, tid in zip(world_xy, tracker_ids):
        px = int((xy[0] - mn[0]) / span[0] * (size - 1))
        py = int((xy[1] - mn[1]) / span[1] * (size - 1))
        cv2.circle(canvas, (px, py), 5, (0, 220, 255), -1)
        cv2.putText(canvas, str(int(tid)), (px + 6, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return canvas
