"""Layer strategico e validazione con dati reali FastF1."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_session(year: int, gp: str, session: str = "R", cache: str = ".fastf1_cache"):
    """Carica una sessione F1 reale (timing + telemetria) via FastF1."""
    import fastf1

    Path(cache).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache)
    ses = fastf1.get_session(year, gp, session)
    ses.load()
    return ses


def stint_summary(session) -> pd.DataFrame:
    """Riassunto degli stint: pilota, mescola, giri, tempo medio."""
    laps = session.laps
    grp = laps.groupby(["Driver", "Stint", "Compound"])
    out = grp.agg(
        laps=("LapNumber", "count"),
        avg_laptime_s=("LapTime", lambda s: s.dt.total_seconds().mean()),
    ).reset_index()
    return out


def tyre_degradation(session, driver: str) -> pd.DataFrame:
    """Regressione lineare del tempo sul giro vs età gomma, per stint."""
    laps = session.laps.pick_drivers(driver).pick_quicklaps()
    df = laps[["Stint", "Compound", "TyreLife", "LapTime"]].copy()
    df["laptime_s"] = df["LapTime"].dt.total_seconds()
    rows = []
    for (stint, compound), g in df.groupby(["Stint", "Compound"]):
        if len(g) >= 3:
            slope = g["laptime_s"].cov(g["TyreLife"]) / g["TyreLife"].var()
            rows.append({"stint": stint, "compound": compound,
                         "deg_s_per_lap": round(float(slope), 3), "laps": len(g)})
    return pd.DataFrame(rows)


def validate_speed(estimated_kmh: pd.Series, session, driver: str) -> dict:
    """Confronta la velocità stimata dal video con la telemetria FastF1.

    Ritorna metriche d'errore (MAE, MAPE) — il cuore della sezione 'Risultati'.
    """
    tel = session.laps.pick_drivers(driver).get_telemetry()
    real = tel["Speed"].to_numpy()
    est = estimated_kmh.to_numpy()
    n = min(len(real), len(est))
    if n == 0:
        return {"mae_kmh": None, "mape_pct": None, "n": 0}
    real, est = real[:n], est[:n]
    mae = float(abs(real - est).mean())
    mape = float((abs(real - est) / abs(real).clip(min=1)).mean() * 100)
    return {"mae_kmh": round(mae, 2), "mape_pct": round(mape, 2), "n": n}
