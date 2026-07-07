"""Simulazione Monte Carlo della strategia di gara sotto incertezza (vettorizzata).

Un muretto box non decide sul tempo "medio" ma sul rischio: Safety Car, variabilita del
degrado e del pit-stop cambiano l'esito. Questo modulo simula migliaia di gare campionando
quelle incertezze e restituisce la DISTRIBUZIONE dei tempi e la probabilita che una
strategia batta un'altra.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strategy import DEFAULT_PIT_LOSS, MIN_STINT, enumerate_strategies, simulate_strategy


@dataclass(frozen=True)
class RaceConditions:
    total_laps: int
    pit_loss: float = DEFAULT_PIT_LOSS
    pit_loss_sd: float = 1.5          # variabilita del tempo ai box (s)
    sc_prob: float = 0.35             # probabilita di almeno una Safety Car
    sc_pit_discount: float = 0.55     # quota di pit-loss risparmiata pittando sotto SC
    deg_noise_sd: float = 0.02        # variabilita del degrado (s/giro)


def _sample_times(stints, tyre_models, cond: RaceConditions, n: int, rng) -> np.ndarray:
    """Vettoriale: n tempi gara campionati per una strategia."""
    lengths = np.array([ln for _, ln in stints], dtype=float)
    tri = lengths * (lengths - 1) / 2.0                      # somma delle eta 0..L-1
    tri2 = lengths * (lengths - 1) * (2 * lengths - 1) / 6.0  # somma dei quadrati delle eta

    det = sum(tyre_models[c].base * ln + tyre_models[c].deg * tri[i]
              + getattr(tyre_models[c], "deg2", 0.0) * tri2[i]
              for i, (c, ln) in enumerate(stints))
    total = np.full(n, det)

    deg_noise = rng.normal(0, cond.deg_noise_sd, size=(n, len(stints)))
    total += (deg_noise * tri).sum(axis=1)

    n_stops = len(stints) - 1
    if n_stops > 0:
        boundaries = np.cumsum(lengths)[:-1]                 # giro di ogni sosta
        pit = rng.normal(cond.pit_loss, cond.pit_loss_sd, size=(n, n_stops))
        sc = rng.random(n) < cond.sc_prob
        sc_lap = rng.integers(1, cond.total_laps + 1, size=n)
        for j, lp in enumerate(boundaries):
            free = sc & (np.abs(lp - sc_lap) <= 3)           # pit coincide con la SC
            pit[:, j] = np.where(free, pit[:, j] * (1 - cond.sc_pit_discount), pit[:, j])
        total += pit.sum(axis=1)
    return total


def simulate_strategy_mc(stints, tyre_models, cond: RaceConditions,
                         n: int = 5000, seed: int = 0) -> dict:
    """Distribuzione del tempo gara di una strategia."""
    t = _sample_times(stints, tyre_models, cond, n, np.random.default_rng(seed))
    return {"mean_s": float(t.mean()), "p10_s": float(np.percentile(t, 10)),
            "p50_s": float(np.percentile(t, 50)), "p90_s": float(np.percentile(t, 90)),
            "std_s": float(t.std())}


def win_probability(stints_a, stints_b, tyre_models, cond: RaceConditions,
                    n: int = 5000, seed: int = 0) -> float:
    """Stima di P(strategia A piu veloce di B) via Monte Carlo."""
    rng = np.random.default_rng(seed)
    ta = _sample_times(stints_a, tyre_models, cond, n, rng)
    tb = _sample_times(stints_b, tyre_models, cond, n, rng)
    return float(np.mean(ta < tb))


def optimize_mc(tyre_models, cond: RaceConditions, compounds, max_stops: int = 2,
                require_two_compounds: bool = True, topk: int = 8,
                n: int = 5000, seed: int = 0) -> dict:
    """Strategia col tempo atteso minimo, robustezza e probabilita di battere la seconda.

    Prima si riduce lo spazio con una classifica deterministica, poi si valutano in
    Monte Carlo solo le migliori candidate (piu efficiente e realistico).
    """
    det = []
    for stints in enumerate_strategies(cond.total_laps, compounds, max_stops, MIN_STINT):
        if require_two_compounds and len({c for c, _ in stints}) < 2:
            continue
        det.append((simulate_strategy(cond.total_laps, stints, tyre_models, cond.pit_loss), stints))
    det.sort(key=lambda t: t[0])
    shortlist = [s for _, s in det[:topk]]

    ranked = [{"stints": s, **simulate_strategy_mc(s, tyre_models, cond, n=n, seed=seed)}
              for s in shortlist]
    ranked.sort(key=lambda d: d["mean_s"])
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else ranked[0]
    p = win_probability(best["stints"], second["stints"], tyre_models, cond, n=n, seed=seed)
    return {"best": best, "second": second, "win_prob_vs_second": p,
            "n_evaluated": len(det), "ranked": ranked}
