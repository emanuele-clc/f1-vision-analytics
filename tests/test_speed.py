"""Test della stima di velocità (nessuna dipendenza CV pesante)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va.speed import SpeedEstimator, _Kalman1D  # noqa: E402


def test_constant_speed():
    # Auto che percorre 10 m/frame a 10 fps -> 100 m/s -> 360 km/h (poi clip a 380)
    est = SpeedEstimator(fps=10.0, smoothing="none", max_speed_kmh=400)
    est.update(np.array([1]), np.array([[0.0, 0.0]]))
    speeds = est.update(np.array([1]), np.array([[10.0, 0.0]]))
    assert abs(speeds[1] - 360.0) < 1e-6


def test_speed_clip():
    est = SpeedEstimator(fps=10.0, smoothing="none", max_speed_kmh=380)
    est.update(np.array([1]), np.array([[0.0, 0.0]]))
    speeds = est.update(np.array([1]), np.array([[50.0, 0.0]]))
    assert speeds[1] == 380.0


def test_kalman_converges():
    kf = _Kalman1D(200.0)
    for _ in range(50):
        out = kf.update(250.0)
    assert 240.0 < out < 250.0


def test_gaps_symmetric_distance():
    est = SpeedEstimator(fps=10.0)
    gaps = est.gaps(np.array([1, 2]), np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert abs(gaps[(1, 2)] - 5.0) < 1e-6
