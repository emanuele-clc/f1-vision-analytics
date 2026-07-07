"""Duello testa a testa: aria sporca, posizione in pista e strategie dei rivali.

Simula due auto giro per giro con i loro piani gomme. La posizione in pista dipende dal
tempo cumulato; quando l'inseguitore e vicino (aria sporca) perde tempo e fatica a passare,
a meno che non abbia un vantaggio di passo (es. gomme fresche dopo un undercut). Emergono
cosi naturalmente undercut e overcut contro un rivale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strategy import DEFAULT_PIT_LOSS, MIN_STINT, enumerate_strategies, simulate_strategy


@dataclass(frozen=True)
class DuelConditions:
    total_laps: int
    pit_loss: float = DEFAULT_PIT_LOSS
    pit_loss_sd: float = 1.0
    lap_noise_sd: float = 0.08       # variabilita del passo per giro (s)
    dirty_air_s: float = 0.45        # tempo perso/giro entro la soglia di distacco
    dirty_air_gap: float = 1.2       # distacco (s) sotto cui si e in aria sporca
    overtake_margin: float = 0.5     # vantaggio di passo/giro per superare in pista
    # --- imprevisti ---
    sc_prob: float = 0.40            # probabilita di una Safety Car
    sc_bunch: float = 0.20           # il distacco si comprime a questa frazione sotto SC
    sc_pit_discount: float = 0.55    # pit-loss risparmiato se si sosta sotto SC
    dnf_prob: float = 0.06           # probabilita di ritiro per ciascuna auto
    rain_prob: float = 0.0           # probabilita di pioggia (caos: rimescola l'ordine)
    rain_noise_mult: float = 6.0     # amplificazione del rumore dopo l'inizio pioggia


def _lap_schedule(stints, tyre_models, total_laps):
    """Tempo base per giro (senza rumore) e insieme dei giri di sosta."""
    times, pit_laps, lap = [], set(), 0
    for i, (compound, length) in enumerate(stints):
        m = tyre_models[compound]
        for age in range(length):
            if lap >= total_laps:
                break
            times.append(m.base + m.deg * age)
            lap += 1
        if i < len(stints) - 1:
            pit_laps.add(lap - 1)               # sosta eseguita dopo questo giro
    while len(times) < total_laps:              # sicurezza
        times.append(times[-1] if times else 90.0)
    return np.array(times[:total_laps]), pit_laps


def duel_win_probability(stints_a, stints_b, tyre_models, cond: DuelConditions,
                         start_gap: float = 0.0, n: int = 3000, seed: int = 0) -> float:
    """P(l'auto A finisce davanti a B), con aria sporca e posizione in pista.

    start_gap > 0 = A parte davanti a B di start_gap secondi.
    """
    rng = np.random.default_rng(seed)
    base_a, pit_a = _lap_schedule(stints_a, tyre_models, cond.total_laps)
    base_b, pit_b = _lap_schedule(stints_b, tyre_models, cond.total_laps)

    # imprevisti campionati a inizio simulazione
    sc_on = rng.random(n) < cond.sc_prob
    sc_lap = rng.integers(1, cond.total_laps + 1, n)
    rain_on = rng.random(n) < cond.rain_prob
    rain_lap = rng.integers(1, cond.total_laps + 1, n)
    a_dnf = rng.random(n) < cond.dnf_prob
    b_dnf = rng.random(n) < cond.dnf_prob

    cum_a = np.zeros(n)
    cum_b = np.full(n, float(start_gap))
    for lap in range(cond.total_laps):
        wet = rain_on & (lap >= rain_lap)                 # pioggia in corso -> caos
        noise = np.where(wet, cond.lap_noise_sd * cond.rain_noise_mult, cond.lap_noise_sd)
        la = base_a[lap] + rng.normal(0, 1, n) * noise
        lb = base_b[lap] + rng.normal(0, 1, n) * noise
        near_sc = sc_on & (np.abs(lap - sc_lap) <= 2)
        if lap in pit_a:
            pit = rng.normal(cond.pit_loss, cond.pit_loss_sd, n)
            la += np.where(near_sc, pit * (1 - cond.sc_pit_discount), pit)
        if lap in pit_b:
            pit = rng.normal(cond.pit_loss, cond.pit_loss_sd, n)
            lb += np.where(near_sc, pit * (1 - cond.sc_pit_discount), pit)
        cum_a += la
        cum_b += lb

        # Safety Car: il gruppo si compatta (il distacco si riduce)
        at_sc = sc_on & (lap == sc_lap)
        if at_sc.any():
            gap0 = cum_b - cum_a
            cum_b = np.where(at_sc, cum_a + gap0 * cond.sc_bunch, cum_b)

        gap = cum_b - cum_a                       # >0 A davanti
        a_ahead = gap > 0
        close = np.abs(gap) < cond.dirty_air_gap
        trailing_lap = np.where(a_ahead, lb, la)
        leader_lap = np.where(a_ahead, la, lb)
        faster = (leader_lap - trailing_lap) > cond.overtake_margin
        stuck = close & ~faster
        cum_b += np.where(stuck & a_ahead, cond.dirty_air_s, 0.0)
        cum_a += np.where(stuck & ~a_ahead, cond.dirty_air_s, 0.0)

    # esito: A davanti se non si ritira e (il rivale si ritira oppure ha tempo minore)
    a_ahead_final = (~a_dnf) & (b_dnf | (cum_a < cum_b))
    return float(np.mean(a_ahead_final))


def best_response(rival_stints, tyre_models, cond: DuelConditions, compounds,
                  start_gap: float = 0.0, max_stops: int = 2,
                  require_two_compounds: bool = True, topk: int = 6,
                  n: int = 2500, seed: int = 0) -> dict:
    """Trova la nostra strategia che massimizza P(finire davanti) al rivale dato.

    Riduce lo spazio con una classifica deterministica, poi valuta il duello sulle migliori.
    Etichetta la scelta come undercut / overcut / passo rispetto al rivale.
    """
    det = []
    for stints in enumerate_strategies(cond.total_laps, compounds, max_stops, MIN_STINT):
        if require_two_compounds and len({c for c, _ in stints}) < 2:
            continue
        det.append((simulate_strategy(cond.total_laps, stints, tyre_models, cond.pit_loss), stints))
    det.sort(key=lambda t: t[0])
    shortlist = [s for _, s in det[:topk]]

    rival_first_pit = _lap_schedule(rival_stints, tyre_models, cond.total_laps)[1]
    rival_pit = min(rival_first_pit) if rival_first_pit else cond.total_laps

    scored = []
    for stints in shortlist:
        p = duel_win_probability(stints, rival_stints, tyre_models, cond,
                                 start_gap=start_gap, n=n, seed=seed)
        our_pit = _lap_schedule(stints, tyre_models, cond.total_laps)[1]
        our_pit = min(our_pit) if our_pit else cond.total_laps
        label = ("undercut" if our_pit < rival_pit - 1
                 else "overcut" if our_pit > rival_pit + 1 else "passo")
        scored.append({"stints": stints, "win_prob": p, "first_pit": our_pit, "label": label})
    scored.sort(key=lambda d: -d["win_prob"])
    return {"best": scored[0], "candidates": scored, "rival_first_pit": rival_pit}
