"""Backtesting: confronta la strategia consigliata dal motore con quella realmente
adottata in gara. E il modo piu concreto per validare un motore di strategia:
"nelle gare passate, avrebbe fatto la chiamata giusta?".
"""
from __future__ import annotations

import pandas as pd

from .features import fuel_corrected_degradation
from .strategy import fit_tyre_models, optimize_strategy, simulate_strategy


def _max_stint_from_laps(laps: pd.DataFrame) -> dict:
    """Durata massima realistica per mescola = giri effettivamente percorsi + margine."""
    life = laps.dropna(subset=["TyreLife"]).groupby(
        laps["Compound"].astype(str).str.upper())["TyreLife"].max()
    return {c: int(v) + 2 for c, v in life.items()}


def _robust_optimize(total, keys, tyre, pit_loss, max_stint):
    """Ottimizza con vincoli realistici; allenta i vincoli solo se non c'e soluzione."""
    for stops, min_st, cap in [(2, 8, max_stint), (3, 8, max_stint),
                               (3, 6, max_stint), (3, 5, None)]:
        try:
            return optimize_strategy(total, keys, tyre, max_stops=stops,
                                     pit_loss=pit_loss, min_stint=min_st, max_stint=cap)
        except ValueError:
            continue
    return optimize_strategy(total, keys, tyre, max_stops=3, pit_loss=pit_loss, min_stint=5)


def driver_stints(laps: pd.DataFrame, driver: str) -> list[tuple[str, int]]:
    """Stint realmente percorsi da un pilota: (mescola, numero di giri)."""
    d = laps[laps["Driver"] == driver].sort_values("LapNumber")
    stints = []
    for _, g in d.groupby("Stint"):
        comp = str(g["Compound"].iloc[0]).upper()
        stints.append((comp, int(len(g))))
    return stints


def backtest_race(laps: pd.DataFrame, total_laps: int | None = None,
                  max_stops: int = 2, pit_loss: float = 22.0) -> dict:
    """Backtest di una gara: strategia consigliata vs reale per ogni pilota.

    Ritorna il piano consigliato, un DataFrame per pilota (piano reale, tempo reale
    stimato, tempo ottimo, delta) e metriche aggregate.
    """
    tyre = fit_tyre_models(fuel_corrected_degradation(laps))
    keys = list(tyre)
    if len(keys) < 2:
        return {"tyre": tyre, "recommended": None, "drivers": pd.DataFrame(), "metrics": {}}
    if total_laps is None:
        total_laps = int(laps["LapNumber"].max())

    max_stint = _max_stint_from_laps(laps)
    recommended = _robust_optimize(total_laps, keys, tyre, pit_loss, max_stint)

    opt_cache: dict[int, float] = {}
    rows = []
    for driver in laps["Driver"].unique():
        actual = driver_stints(laps, driver)
        run = sum(n for _, n in actual)
        if run < total_laps * 0.8:                       # scarta ritirati / doppiati
            continue
        if any(c not in tyre for c in {c for c, _ in actual}):
            continue
        if len({c for c, _ in actual}) < 2:              # regola due mescole (gara asciutta)
            continue
        actual_time = simulate_strategy(run, actual, tyre, pit_loss=pit_loss)
        if run not in opt_cache:
            opt_cache[run] = _robust_optimize(run, keys, tyre, pit_loss, max_stint).total_time_s
        opt_time = opt_cache[run]
        rows.append({
            "driver": driver,
            "actual_plan": " -> ".join(f"{c}({n})" for c, n in actual),
            "actual_stops": len(actual) - 1,
            "actual_time_s": round(actual_time, 1),
            "optimal_time_s": round(opt_time, 1),
            "delta_s": round(actual_time - opt_time, 1),   # >0 = tempo lasciato sul tavolo
        })
    df = pd.DataFrame(rows).sort_values("delta_s").reset_index(drop=True)

    metrics = {}
    if not df.empty:
        rec_stops = recommended.n_stops
        metrics = {
            "drivers": len(df),
            "match_stops_pct": round((df["actual_stops"] == rec_stops).mean() * 100, 1),
            "median_delta_s": round(float(df["delta_s"].median()), 2),
            "best_actual_delta_s": round(float(df["delta_s"].min()), 2),
        }
    return {"tyre": tyre, "recommended": recommended, "drivers": df, "metrics": metrics}


def backtest_sessions(sessions: list[tuple[str, pd.DataFrame]], max_stops: int = 2) -> pd.DataFrame:
    """Backtest su piu gare: una riga di riepilogo per gara."""
    rows = []
    for label, laps in sessions:
        res = backtest_race(laps, max_stops=max_stops)
        m = res["metrics"]
        if not m:
            continue
        rec = res["recommended"]
        rows.append({
            "gara": label,
            "consigliata": " -> ".join(f"{c}({n})" for c, n in rec.stints),
            "match_soste_%": m["match_stops_pct"],
            "delta_mediano_s": m["median_delta_s"],
            "piloti": m["drivers"],
        })
    return pd.DataFrame(rows)
