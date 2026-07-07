"""Test del feature engineering (senza rete, con dati sintetici)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import features  # noqa: E402


def _synthetic_laps():
    rows = []
    for life in range(1, 11):
        rows.append({"Compound": "SOFT", "TyreLife": life, "LapNumber": life,
                     "Stint": 1, "laptime_s": 90.0 + 0.5 * life})
        rows.append({"Compound": "HARD", "TyreLife": life, "LapNumber": 10 + life,
                     "Stint": 2, "laptime_s": 91.0 + 0.1 * life})
    return pd.DataFrame(rows)


def test_build_laptime_dataset_shapes():
    x, y = features.build_laptime_dataset(_synthetic_laps())
    assert len(x) == len(y) == 20
    assert "tyre_life" in x.columns and "comp_SOFT" in x.columns


def test_degradation_recovers_slope():
    deg = features.degradation_table(_synthetic_laps())
    soft = deg[deg["compound"] == "SOFT"].iloc[0]
    hard = deg[deg["compound"] == "HARD"].iloc[0]
    assert soft["deg_s_per_lap"] == pytest.approx(0.5, abs=0.05)
    assert hard["deg_s_per_lap"] == pytest.approx(0.1, abs=0.05)


def _laps_with_fuel():
    """SOFT in due stint (giri diversi) + HARD: rende l'effetto carburante identificabile."""
    fuel = -0.03
    rows = []

    def add(comp, stint, lap0, lives, deg, base=90.0):
        lap = lap0
        for life in lives:
            rows.append({"Compound": comp, "TyreLife": life, "LapNumber": lap,
                         "Stint": stint, "laptime_s": base + fuel * lap + deg * life})
            lap += 1

    add("SOFT", 1, 1, range(1, 11), 0.10)
    add("HARD", 2, 11, range(1, 16), 0.04)
    add("SOFT", 3, 26, range(1, 14), 0.10)
    return pd.DataFrame(rows)


def test_fuel_correction_separates_effects():
    tab = features.fuel_corrected_degradation(_laps_with_fuel())
    soft = tab[tab["compound"] == "SOFT"].iloc[0]
    hard = tab[tab["compound"] == "HARD"].iloc[0]
    assert soft["fuel_s_per_lap"] == pytest.approx(-0.03, abs=0.01)
    assert soft["deg_s_per_lap"] == pytest.approx(0.10, abs=0.02)
    assert hard["deg_s_per_lap"] == pytest.approx(0.04, abs=0.02)
