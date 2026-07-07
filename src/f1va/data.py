"""Data layer: caricamento sessioni F1 reali via FastF1 e pulizia dei giri.

FastF1 espone timing e telemetria ufficiali. La prima chiamata scarica i dati,
le successive usano la cache locale.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_session(year: int, gp: str, session: str = "R", cache: str = ".fastf1_cache"):
    """Carica una sessione (R=gara, Q=qualifica, FP1..3). Ritorna l'oggetto FastF1."""
    import fastf1

    Path(cache).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache)
    ses = fastf1.get_session(year, gp, session)
    ses.load()
    return ses


def laps_dataframe(session) -> pd.DataFrame:
    """Estrae un DataFrame di giri pulito e comodo per analisi/ML.

    Colonne: Driver, Team, LapNumber, Stint, Compound, TyreLife, LapTime (s),
    Position, Sector times (s), IsPitLap.
    """
    laps = session.laps.copy()
    df = pd.DataFrame({
        "Driver": laps["Driver"],
        "Team": laps.get("Team"),
        "LapNumber": laps["LapNumber"],
        "Stint": laps["Stint"],
        "Compound": laps["Compound"],
        "TyreLife": laps["TyreLife"],
        "Position": laps.get("Position"),
        "laptime_s": laps["LapTime"].dt.total_seconds(),
        "s1_s": laps["Sector1Time"].dt.total_seconds(),
        "s2_s": laps["Sector2Time"].dt.total_seconds(),
        "s3_s": laps["Sector3Time"].dt.total_seconds(),
    })
    df["is_pit_lap"] = laps["PitInTime"].notna() | laps["PitOutTime"].notna()
    return df.reset_index(drop=True)


def quicklaps(df: pd.DataFrame, threshold: float = 1.07) -> pd.DataFrame:
    """Filtra i giri 'validi' (rimuove out/in-lap e giri lenti oltre soglia * best)."""
    clean = df[(~df["is_pit_lap"]) & df["laptime_s"].notna()].copy()
    if clean.empty:
        return clean
    best = clean["laptime_s"].min()
    return clean[clean["laptime_s"] <= best * threshold].reset_index(drop=True)
