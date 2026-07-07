"""Replay: ricostruzione frame-by-frame delle posizioni in pista (stile race-replay).

Usa i dati di posizione (X, Y) della telemetria FastF1 per generare, a intervalli
regolari, la posizione di ogni pilota. Il risultato alimenta la visualizzazione.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def track_outline(session) -> pd.DataFrame:
    """Polyline del tracciato (X, Y) dal giro piu veloce, per disegnare il circuito."""
    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()
    return pd.DataFrame({"x": pos["X"].to_numpy(), "y": pos["Y"].to_numpy()})


def build_frames(session, step_s: float = 1.0) -> list[dict]:
    """Genera una lista di frame: per ogni istante, posizioni di tutti i piloti.

    step_s = risoluzione temporale in secondi. Ogni frame:
        {"t": float, "cars": {driver: {"x": .., "y": ..}}}
    """
    drivers = session.drivers
    tracks: dict[str, pd.DataFrame] = {}
    t_max = 0.0
    for drv in drivers:
        try:
            laps = session.laps.pick_drivers(drv)
            pos = laps.get_pos_data()
        except Exception:
            continue
        if pos is None or len(pos) == 0:
            continue
        abbr = session.get_driver(drv)["Abbreviation"]
        secs = (pos["Time"] - pos["Time"].iloc[0]).dt.total_seconds().to_numpy()
        tracks[abbr] = pd.DataFrame({"t": secs, "x": pos["X"].to_numpy(), "y": pos["Y"].to_numpy()})
        t_max = max(t_max, float(secs[-1]))

    frames = []
    for t in np.arange(0, t_max, step_s):
        cars = {}
        for abbr, tr in tracks.items():
            x = float(np.interp(t, tr["t"], tr["x"]))
            y = float(np.interp(t, tr["t"], tr["y"]))
            cars[abbr] = {"x": x, "y": y}
        frames.append({"t": float(t), "cars": cars})
    return frames
