"""Feature engineering: dai giri puliti ai dataset per il machine learning."""
from __future__ import annotations

import numpy as np
import pandas as pd

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def build_laptime_dataset(laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Costruisce (X, y) per la previsione del tempo sul giro.

    Feature: eta gomma (TyreLife), numero giro (proxy del carburante), numero stint,
    e mescola in one-hot. Target: laptime_s.
    """
    df = laps.dropna(subset=["laptime_s", "TyreLife"]).copy()
    x = pd.DataFrame({
        "tyre_life": df["TyreLife"].astype(float),
        "lap_number": df["LapNumber"].astype(float),
        "stint": df["Stint"].astype(float),
    })
    for c in COMPOUNDS:
        x[f"comp_{c}"] = (df["Compound"].astype(str).str.upper() == c).astype(int)
    y = df["laptime_s"].astype(float).reset_index(drop=True)
    return x.reset_index(drop=True), y


def degradation_table(laps: pd.DataFrame, min_laps: int = 4) -> pd.DataFrame:
    """Stima il degrado (secondi/giro) per mescola con una regressione lineare.

    Ritorna: compound, base_s (tempo a gomma nuova), deg_s_per_lap, laps.
    """
    rows = []
    df = laps.dropna(subset=["laptime_s", "TyreLife"])
    for compound, g in df.groupby(df["Compound"].astype(str).str.upper()):
        if len(g) < min_laps or g["TyreLife"].nunique() < 2:
            continue
        life = g["TyreLife"].to_numpy(dtype=float)
        lap = g["laptime_s"].to_numpy(dtype=float)
        deg, base = np.polyfit(life, lap, 1)  # lap = deg*life + base
        rows.append({
            "compound": compound,
            "base_s": round(float(base), 3),
            "deg_s_per_lap": round(float(deg), 4),
            "laps": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values("compound").reset_index(drop=True)
