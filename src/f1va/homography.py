"""Omografia: proiezione dei punti immagine sulla mappa 2D del circuito."""
from __future__ import annotations

import cv2
import numpy as np


class TrackProjector:
    """Stima l'omografia da corrispondenze immagine<->mappa e proietta i punti.

    Le corrispondenze vanno calibrate per ogni inquadratura/circuito: bastano 4+
    punti di cui si conoscono le coordinate sia in pixel sia sulla mappa del tracciato.
    """

    def __init__(self, image_points, world_points, meters_per_world_unit: float = 1.0):
        img = np.asarray(image_points, dtype=np.float32)
        wrl = np.asarray(world_points, dtype=np.float32)
        if len(img) < 4 or len(img) != len(wrl):
            raise ValueError("Servono >=4 corrispondenze immagine/mappa della stessa lunghezza.")
        self.H, _ = cv2.findHomography(img, wrl, method=cv2.RANSAC)
        self.meters_per_world_unit = meters_per_world_unit

    def project(self, points_px: np.ndarray) -> np.ndarray:
        """Proietta (N,2) punti immagine -> (N,2) coordinate mappa (in metri)."""
        if points_px.size == 0:
            return np.empty((0, 2))
        pts = points_px.reshape(-1, 1, 2).astype(np.float32)
        world = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        return world * self.meters_per_world_unit
