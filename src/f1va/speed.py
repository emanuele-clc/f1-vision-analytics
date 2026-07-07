"""Stima di velocità e gap dalle posizioni proiettate sulla mappa (metri)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class SpeedEstimator:
    """Mantiene lo storico per track_id e stima la velocità (km/h).

    Un filtro di Kalman a velocità costante smorza il rumore della detection.
    Semplificato a 1D sul modulo dello spostamento; per la produzione conviene
    un Kalman 2D (stato = [x, y, vx, vy]).
    """

    def __init__(self, fps: float, smoothing: str = "kalman", max_speed_kmh: float = 380):
        self.fps = fps
        self.dt = 1.0 / fps
        self.smoothing = smoothing
        self.max_speed_kmh = max_speed_kmh
        self._last_pos: dict[int, np.ndarray] = {}
        self._kf: dict[int, "_Kalman1D"] = {}
        self.history: dict[int, list[float]] = defaultdict(list)

    def update(self, tracker_ids: np.ndarray, world_xy: np.ndarray) -> dict[int, float]:
        out: dict[int, float] = {}
        for tid, pos in zip(tracker_ids, world_xy):
            tid = int(tid)
            if tid in self._last_pos:
                dist_m = float(np.linalg.norm(pos - self._last_pos[tid]))
                speed_ms = dist_m / self.dt
                speed_kmh = min(speed_ms * 3.6, self.max_speed_kmh)
                if self.smoothing == "kalman":
                    kf = self._kf.setdefault(tid, _Kalman1D(speed_kmh))
                    speed_kmh = kf.update(speed_kmh)
                out[tid] = speed_kmh
                self.history[tid].append(speed_kmh)
            self._last_pos[tid] = pos
        return out

    def gaps(self, tracker_ids: np.ndarray, world_xy: np.ndarray) -> dict[tuple[int, int], float]:
        """Distanza euclidea (m) tra coppie di auto — proxy per il gap sulla mappa."""
        gaps: dict[tuple[int, int], float] = {}
        ids = [int(t) for t in tracker_ids]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                gaps[(ids[i], ids[j])] = float(np.linalg.norm(world_xy[i] - world_xy[j]))
        return gaps


class _Kalman1D:
    """Filtro di Kalman scalare a velocità costante (smoothing 1D)."""

    def __init__(self, x0: float, q: float = 4.0, r: float = 25.0):
        self.x = x0
        self.p = 1.0
        self.q = q  # rumore di processo
        self.r = r  # rumore di misura

    def update(self, z: float) -> float:
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1 - k)
        return self.x
