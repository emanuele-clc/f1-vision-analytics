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
    life = df["TyreLife"].astype(float)
    x = pd.DataFrame({
        "tyre_life": life,
        "tyre_life_sq": life**2,               # cattura il degrado non lineare
        "lap_number": df["LapNumber"].astype(float),
        "stint": df["Stint"].astype(float),
    })
    for c in COMPOUNDS:
        x[f"comp_{c}"] = (df["Compound"].astype(str).str.upper() == c).astype(int)
    for opt in ("TrackTemp", "AirTemp"):       # feature opzionali se presenti nei dati
        if opt in df.columns and df[opt].notna().any():
            x[opt.lower()] = df[opt].astype(float)
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


def fuel_corrected_degradation(laps: pd.DataFrame, min_laps: int = 4) -> pd.DataFrame:
    """Degrado gomma separato dall'effetto carburante.

    Il tempo sul giro cala col carburante che si consuma (auto piu leggera) e cresce
    con l'usura gomma. Un modello a minimi quadrati stima insieme, per ogni mescola,
    `base` e `degrado`, piu un unico coefficiente globale `carburante` (s per giro di
    carico residuo, tipicamente negativo). L'effetto carburante e identificabile quando
    una mescola compare in stint a intervalli di giri diversi.

    Ritorna: compound, base_s, deg_s_per_lap (corretto), fuel_s_per_lap, laps.
    """
    df = laps.dropna(subset=["laptime_s", "TyreLife", "LapNumber"]).copy()
    comp = df["Compound"].astype(str).str.upper()
    compounds = [c for c in sorted(comp.unique())
                 if (comp == c).sum() >= min_laps and df.loc[comp == c, "TyreLife"].nunique() >= 2]
    if not compounds:
        return pd.DataFrame(columns=["compound", "base_s", "deg_s_per_lap", "fuel_s_per_lap", "laps"])

    mask = comp.isin(compounds)
    df, comp = df[mask], comp[mask]
    life = df["TyreLife"].to_numpy(float)
    lap = df["LapNumber"].to_numpy(float)

    cols, names = [], []
    for c in compounds:                      # base + degrado per mescola (interazioni)
        ind = (comp == c).to_numpy(float)
        cols += [ind, ind * life]
        names += [("base", c), ("deg", c)]
    cols.append(lap)                         # unico termine carburante globale
    names.append(("fuel", None))

    design = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(design, df["laptime_s"].to_numpy(float), rcond=None)
    fuel = float(coef[names.index(("fuel", None))])

    rows = []
    for c in compounds:
        rows.append({
            "compound": c,
            "base_s": round(float(coef[names.index(("base", c))]), 3),
            "deg_s_per_lap": round(float(coef[names.index(("deg", c))]), 4),
            "fuel_s_per_lap": round(fuel, 4),
            "laps": int((comp == c).sum()),
        })
    return pd.DataFrame(rows).reset_index(drop=True)
