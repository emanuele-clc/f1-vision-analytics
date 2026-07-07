"""Strategia live: ricalcola la strategia ottimale rimanente durante la gara.

Dato lo stato attuale (giro, mescola e usura correnti, mescole gia usate), trova il
piano rimanente col tempo minimo e traduce la decisione immediata: restare fuori o
andare ai box, quando e con quale mescola.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product


@dataclass(frozen=True)
class RaceState:
    total_laps: int
    current_lap: int                 # giri gia completati
    current_compound: str
    current_tyre_age: int            # eta della gomma attuale
    used_compounds: frozenset = field(default_factory=frozenset)
    pit_loss: float = 22.0


@dataclass(frozen=True)
class LivePlan:
    stay_out_laps: int               # giri ancora sulla gomma attuale
    next_stints: tuple               # ((compound, laps), ...) dopo la sosta
    remaining_time_s: float
    action: str


def _splits_first(total: int, parts: int, min_rest: int):
    """Compositions: prima parte >=1, le altre >= min_rest."""
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - min_rest * (parts - 1) + 1):
        for rest in _compositions(total - first, parts - 1, min_rest):
            yield (first, *rest)


def _compositions(total: int, parts: int, min_each: int):
    if parts == 0:
        if total == 0:
            yield ()
        return
    if parts == 1:
        if total >= min_each:
            yield (total,)
        return
    for first in range(min_each, total - min_each * (parts - 1) + 1):
        for rest in _compositions(total - first, parts - 1, min_each):
            yield (first, *rest)


def _cost(state: RaceState, stay: int, new_stints, tyre_models) -> float:
    m = tyre_models[state.current_compound]
    t = sum(m.base + m.deg * a
            for a in range(state.current_tyre_age, state.current_tyre_age + stay))
    for compound, length in new_stints:
        mm = tyre_models[compound]
        t += sum(mm.base + mm.deg * a for a in range(length))
    return t + state.pit_loss * len(new_stints)


def recommend(state: RaceState, tyre_models, compounds, max_further_stops: int = 2,
              min_stint: int = 5) -> LivePlan:
    """Piano rimanente col tempo minimo, con la decisione immediata."""
    remaining = state.total_laps - state.current_lap
    best = None
    for k in range(0, max_further_stops + 1):
        for parts in _splits_first(remaining, k + 1, min_stint):
            stay, new_lens = parts[0], parts[1:]
            for combo in product(compounds, repeat=k):
                seq = [state.current_compound, *combo]
                if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
                    continue
                if len(state.used_compounds.union(seq)) < 2:
                    continue
                new_stints = tuple(zip(combo, new_lens))
                c = _cost(state, stay, new_stints, tyre_models)
                if best is None or c < best[0]:
                    best = (c, stay, new_stints)
    if best is None:
        raise ValueError("Nessun piano valido: controlla vincoli e mescole disponibili.")

    cost, stay, new_stints = best
    if not new_stints:
        action = "RESTA FUORI fino alla fine (nessuna sosta conviene)"
    elif stay <= 1:
        action = f"BOX ORA → monta {new_stints[0][0]}"
    else:
        pit_lap = state.current_lap + stay
        action = (f"Resta fuori {stay} giri, poi BOX al giro {pit_lap} "
                  f"→ {new_stints[0][0]}")
    return LivePlan(stay_out_laps=stay, next_stints=new_stints,
                    remaining_time_s=cost, action=action)
