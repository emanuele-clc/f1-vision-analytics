"""Planning: modello gomme, simulatore di gara e ottimizzatore di pit-stop.

Tutta la logica e pura (numeri dentro, numeri fuori): testabile senza rete e
riutilizzabile sia con parametri stimati dai dati reali sia con valori ipotetici.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_PIT_LOSS = 22.0  # secondi persi per un pit-stop (pit lane + sosta)
MIN_STINT = 5            # giri minimi credibili per uno stint


@dataclass(frozen=True)
class TyreModel:
    """Tempo sul giro modellato come base + degrado * eta_gomma (secondi)."""

    base: float
    deg: float

    def stint_time(self, length: int) -> float:
        """Tempo totale di uno stint di `length` giri (eta da 0 a length-1)."""
        return sum(self.base + self.deg * age for age in range(length))


@dataclass(frozen=True)
class StrategyResult:
    stints: tuple
    total_time_s: float
    n_stops: int


def simulate_strategy(total_laps, stints, tyre_models, pit_loss=DEFAULT_PIT_LOSS) -> float:
    """Tempo gara totale per una strategia = somma stint + soste.

    stints: lista di (compound, length). La somma delle length deve fare total_laps.
    """
    if sum(length for _, length in stints) != total_laps:
        raise ValueError("La somma dei giri degli stint deve essere pari a total_laps.")
    total = 0.0
    for i, (compound, length) in enumerate(stints):
        total += tyre_models[compound].stint_time(length)
        if i < len(stints) - 1:
            total += pit_loss
    return total


def _compositions(total: int, parts: int, min_each: int):
    """Genera tutte le suddivisioni di `total` in `parts` interi >= min_each."""
    if parts == 1:
        if total >= min_each:
            yield (total,)
        return
    for first in range(min_each, total - min_each * (parts - 1) + 1):
        for rest in _compositions(total - first, parts - 1, min_each):
            yield (first, *rest)


def enumerate_strategies(total_laps, compounds, max_stops, min_stint=MIN_STINT):
    """Genera strategie plausibili con 1..max_stops soste."""
    from itertools import product

    for n_stops in range(1, max_stops + 1):
        n_stints = n_stops + 1
        for lengths in _compositions(total_laps, n_stints, min_stint):
            for combo in product(compounds, repeat=n_stints):
                # evita due stint consecutivi con la stessa mescola (inutile)
                if any(combo[i] == combo[i + 1] for i in range(len(combo) - 1)):
                    continue
                yield tuple(zip(combo, lengths))


def optimize_strategy(total_laps, compounds, tyre_models, max_stops=2,
                      pit_loss=DEFAULT_PIT_LOSS, require_two_compounds=True,
                      min_stint=MIN_STINT) -> StrategyResult:
    """Trova la strategia col tempo gara minimo (ricerca esaustiva sui piani plausibili)."""
    best: StrategyResult | None = None
    for stints in enumerate_strategies(total_laps, compounds, max_stops, min_stint):
        if require_two_compounds and len({c for c, _ in stints}) < 2:
            continue
        t = simulate_strategy(total_laps, stints, tyre_models, pit_loss)
        if best is None or t < best.total_time_s:
            best = StrategyResult(stints=stints, total_time_s=t, n_stops=len(stints) - 1)
    if best is None:
        raise ValueError("Nessuna strategia valida con i vincoli dati.")
    return best


def fit_tyre_models(deg_table: pd.DataFrame) -> dict[str, TyreModel]:
    """Converte la tabella di degrado (features.degradation_table) in TyreModel per mescola."""
    return {
        row["compound"]: TyreModel(base=float(row["base_s"]), deg=float(row["deg_s_per_lap"]))
        for _, row in deg_table.iterrows()
    }


def undercut_gain(tyre_models, compound_fresh, laps_ahead=2,
                  rival_deg_compound=None, pit_loss=DEFAULT_PIT_LOSS) -> float:
    """Guadagno teorico (s) di un undercut: montare gomma nuova ora vs restare fuori.

    Positivo = l'undercut conviene. Confronta i prossimi `laps_ahead` giri su gomma
    nuova (piu la sosta) contro il rivale che resta su gomma vecchia che degrada.
    """
    fresh = tyre_models[compound_fresh]
    rival = tyre_models[rival_deg_compound or compound_fresh]
    mine = sum(fresh.base + fresh.deg * a for a in range(laps_ahead)) + pit_loss
    theirs = sum(rival.base + rival.deg * (10 + a) for a in range(laps_ahead)) + pit_loss
    return round(theirs - mine, 3)
