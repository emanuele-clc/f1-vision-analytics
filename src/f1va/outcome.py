"""Previsione dell'esito di gara: posizione finale e piazzamento a punti.

Genera esiti di stagione realistici (prestazione auto per team stabile, forma piloti,
qualifica -> griglia, gara con rumore e ritiri) e costruisce un dataset con feature
NOTE prima della gara: griglia di partenza e forma stagionale di pilota e team.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_race_outcomes(n_races: int = 16, n_drivers: int = 20, seed: int = 0,
                           dnf_prob: float = 0.07) -> pd.DataFrame:
    """Esiti sintetici: una riga per pilota per gara, con griglia e posizione finale."""
    rng = np.random.default_rng(seed)
    n_teams = n_drivers // 2
    car = rng.normal(0, 1.0, n_teams)          # prestazione auto (bassa = veloce), stabile
    skill = rng.normal(0, 0.5, n_drivers)      # abilita del pilota, STABILE nella stagione
    rows = []
    for r in range(n_races):
        weekend = rng.normal(0, 0.30, n_drivers)   # forma del singolo weekend
        base = np.array([car[d // 2] + skill[d] + weekend[d] for d in range(n_drivers)])
        quali = base + rng.normal(0, 0.20, n_drivers)
        grid = np.argsort(np.argsort(quali)) + 1
        race = base + rng.normal(0, 0.55, n_drivers)
        dnf = rng.random(n_drivers) < dnf_prob
        race = np.where(dnf, 1e6, race)        # i ritiri finiscono in fondo
        finish = np.argsort(np.argsort(race)) + 1
        for d in range(n_drivers):
            rows.append({
                "Race": r, "Driver": f"D{d + 1:02d}", "Team": f"T{d // 2 + 1:02d}",
                "grid_pos": int(grid[d]), "dnf": bool(dnf[d]), "finish_pos": int(finish[d]),
            })
    return pd.DataFrame(rows)


def build_outcome_dataset(results: pd.DataFrame):
    """Costruisce (X, y) usando solo feature note prima della gara.

    Feature: posizione di partenza e forma stagionale (media dei piazzamenti nelle gare
    precedenti) di pilota e team. Target: posizione finale.
    """
    df = results.sort_values("Race").copy()

    def prior(group_col):
        return (df.groupby(group_col)["finish_pos"]
                .transform(lambda s: s.shift().expanding().mean()))

    df["driver_form"] = prior("Driver")
    df["team_form"] = prior("Team")
    df["driver_form"] = df["driver_form"].fillna(df["grid_pos"])
    df["team_form"] = df["team_form"].fillna(df["grid_pos"])

    x = df[["grid_pos", "driver_form", "team_form"]].astype(float).reset_index(drop=True)
    y = df["finish_pos"].astype(int).reset_index(drop=True)
    return x, y, df.reset_index(drop=True)
