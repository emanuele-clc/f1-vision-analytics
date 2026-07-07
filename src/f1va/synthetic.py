"""Generatore di dati di gara realistici (senza rete).

Serve per allenare e validare i modelli ML anche senza FastF1. Riproduce i fenomeni
chiave: passo base per mescola, degrado gomma NON lineare, sensibilita alla temperatura
pista, effetto carburante, abilita del pilota/team e rumore casuale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# base pace (s), degrado lineare (s/giro), degrado quadratico (s/giro^2)
COMPOUNDS = {
    "SOFT": (88.0, 0.090, 0.0045),
    "MEDIUM": (88.6, 0.050, 0.0022),
    "HARD": (89.3, 0.028, 0.0011),
}
FUEL_S_PER_LAP = 0.035     # secondi in piu per giro di carburante residuo
TEMP_SENS = 0.012          # sensibilita del degrado alla temperatura pista


def _random_splits(total: int, parts: int, rng, min_len: int = 8) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.integers(min_len, total - min_len, size=parts - 1))
    bounds = [0, *cuts, total]
    return [max(min_len, bounds[i + 1] - bounds[i]) for i in range(parts)]


def generate_race(n_drivers: int = 20, laps: int = 53, seed: int = 0,
                  track_temp: float = 30.0) -> pd.DataFrame:
    """Genera i giri di una gara sintetica (DataFrame compatibile con features/quicklaps)."""
    rng = np.random.default_rng(seed)
    comp_names = list(COMPOUNDS)
    temp_factor = 1.0 + TEMP_SENS * (track_temp - 30.0)
    rows = []
    for d in range(n_drivers):
        driver = f"D{d + 1:02d}"
        team = f"T{d // 2 + 1:02d}"
        skill = rng.normal(0, 0.25)                 # passo del pilota
        team_pace = rng.normal(0, 0.20)             # passo del team
        n_stops = int(rng.integers(1, 3))
        splits = _random_splits(laps, n_stops + 1, rng)
        choice = list(rng.choice(comp_names, size=len(splits), replace=True))
        if len(set(choice)) < 2:
            choice[-1] = rng.choice([c for c in comp_names if c != choice[0]])
        lapnum, stint = 1, 1
        for length, comp in zip(splits, choice):
            base, deg, deg2 = COMPOUNDS[comp]
            for age in range(length):
                if lapnum > laps:
                    break
                wear = (deg * age + deg2 * age**2) * temp_factor
                fuel = FUEL_S_PER_LAP * (laps - lapnum)
                lt = base + wear + fuel + skill + team_pace + rng.normal(0, 0.13)
                rows.append({
                    "Driver": driver, "Team": team, "Compound": comp,
                    "TyreLife": age + 1, "LapNumber": lapnum, "Stint": stint,
                    "TrackTemp": round(track_temp, 1), "laptime_s": round(lt, 3),
                    "is_pit_lap": age == length - 1 and stint <= n_stops,
                })
                lapnum += 1
            stint += 1
    return pd.DataFrame(rows)


def generate_season(n_races: int = 10, n_drivers: int = 20, seed: int = 0) -> pd.DataFrame:
    """Genera piu gare con condizioni diverse: dataset ampio e vario per il ML."""
    rng = np.random.default_rng(seed)
    frames = []
    for r in range(n_races):
        laps = int(rng.integers(44, 72))
        temp = float(rng.uniform(18, 46))
        df = generate_race(n_drivers=n_drivers, laps=laps, seed=seed * 1000 + r, track_temp=temp)
        df["Race"] = r
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
