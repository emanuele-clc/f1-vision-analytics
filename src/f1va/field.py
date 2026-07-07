"""Simulazione a campo pieno: tutte le auto insieme, con posizione in pista.

Generalizza il duello a N auto. Ogni auto ha la sua strategia; la posizione dipende dal
tempo cumulato, chi segue da vicino perde tempo in aria sporca e fatica a passare.
Monte Carlo su Safety Car, ritiri e pioggia. Restituisce la distribuzione delle
posizioni finali per ogni auto.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .duel import _lap_schedule


@dataclass(frozen=True)
class FieldConditions:
    total_laps: int
    pit_loss: float = 22.0
    pit_loss_sd: float = 1.0
    lap_noise_sd: float = 0.08
    dirty_air_s: float = 0.40
    dirty_air_gap: float = 1.0
    overtake_margin: float = 0.5
    grid_gap: float = 0.35            # distacco iniziale per posizione in griglia (s)
    sc_prob: float = 0.40
    sc_bunch: float = 0.25
    dnf_prob: float = 0.05
    rain_prob: float = 0.0
    rain_noise_mult: float = 6.0


def simulate_field(strategies: dict, tyre_models, cond: FieldConditions,
                   grid: list | None = None, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Simula la gara per tutte le auto e riassume le posizioni finali.

    strategies: {nome_auto: [(mescola, giri), ...]}.
    grid: ordine di partenza (lista di nomi); se assente, ordine del dizionario.
    """
    cars = list(strategies)
    ncar = len(cars)
    grid = grid or cars
    pos_in_grid = {name: i for i, name in enumerate(grid)}

    scheds = [_lap_schedule(strategies[c], tyre_models, cond.total_laps) for c in cars]
    base = np.stack([s[0] for s in scheds], axis=1)          # (laps, ncar)
    pit_sets = [s[1] for s in scheds]

    rng = np.random.default_rng(seed)
    cum = np.zeros((n, ncar))
    for j, c in enumerate(cars):
        cum[:, j] = pos_in_grid[c] * cond.grid_gap           # gap di partenza

    sc_on = rng.random(n) < cond.sc_prob
    sc_lap = rng.integers(1, cond.total_laps + 1, n)
    rain_on = rng.random(n) < cond.rain_prob
    rain_lap = rng.integers(1, cond.total_laps + 1, n)
    dnf = rng.random((n, ncar)) < cond.dnf_prob

    for lap in range(cond.total_laps):
        wet = rain_on & (lap >= rain_lap)
        noise = np.where(wet, cond.lap_noise_sd * cond.rain_noise_mult, cond.lap_noise_sd)
        lap_t = base[lap][None, :] + rng.normal(0, 1, (n, ncar)) * noise[:, None]
        near_sc = sc_on & (np.abs(lap - sc_lap) <= 2)
        for j in range(ncar):
            if lap in pit_sets[j]:
                pit = rng.normal(cond.pit_loss, cond.pit_loss_sd, n)
                lap_t[:, j] += np.where(near_sc, pit * 0.45, pit)
        cum += lap_t

        at_sc = sc_on & (lap == sc_lap)                      # Safety Car: gruppo compattato
        if at_sc.any():
            leader = cum.min(axis=1, keepdims=True)
            comp = leader + (cum - leader) * cond.sc_bunch
            cum = np.where(at_sc[:, None], comp, cum)

        idx = np.argsort(cum, axis=1)                        # ordine in pista
        scum = np.take_along_axis(cum, idx, axis=1)
        slap = np.take_along_axis(lap_t, idx, axis=1)
        gaps = np.diff(scum, axis=1)
        faster = (slap[:, :-1] - slap[:, 1:]) > cond.overtake_margin
        stuck = (gaps < cond.dirty_air_gap) & ~faster        # inseguitore bloccato
        pen = np.zeros_like(cum)
        np.put_along_axis(pen, idx[:, 1:], np.where(stuck, cond.dirty_air_s, 0.0), axis=1)
        cum += pen

    cum = cum + dnf * 1e6                                     # i ritiri finiscono in fondo
    finish = np.argsort(np.argsort(cum, axis=1), axis=1) + 1  # (n, ncar) posizione finale

    rows = []
    for j, c in enumerate(cars):
        p = finish[:, j]
        rows.append({
            "auto": c,
            "griglia": pos_in_grid[c] + 1,
            "pos_attesa": round(float(p.mean()), 2),
            "P_vittoria_%": round(float(np.mean(p == 1)) * 100, 1),
            "P_podio_%": round(float(np.mean(p <= 3)) * 100, 1),
            "P_punti_%": round(float(np.mean(p <= 10)) * 100, 1),
        })
    return pd.DataFrame(rows).sort_values("pos_attesa").reset_index(drop=True)
