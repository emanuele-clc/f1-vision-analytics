"""Test della simulazione a campo pieno."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va.field import FieldConditions, simulate_field  # noqa: E402
from f1va.strategy import TyreModel  # noqa: E402

TYRE = {"SOFT": TyreModel(78.0, 0.14), "MEDIUM": TyreModel(78.6, 0.09),
        "HARD": TyreModel(79.3, 0.05)}


def _grid(n):
    return {f"C{i:02d}": [("SOFT", 26), ("HARD", 27)] for i in range(n)}


def test_positions_are_a_permutation_on_average():
    strat = _grid(10)
    cond = FieldConditions(53)
    df = simulate_field(strat, TYRE, cond, n=800)
    # la media delle posizioni attese deve essere (N+1)/2
    assert abs(df["pos_attesa"].mean() - 5.5) < 0.2
    assert (df["P_punti_%"] <= 100).all()


def test_pole_beats_backmarker_with_dirty_air():
    strat = _grid(12)
    grid = list(strat)                       # C00 in pole, C11 ultimo
    cond = FieldConditions(53, sc_prob=0, dnf_prob=0, rain_prob=0)
    df = simulate_field(strat, TYRE, cond, grid=grid, n=1500).set_index("auto")
    assert df.loc["C00", "pos_attesa"] < df.loc["C11", "pos_attesa"]


def test_faster_strategy_wins_more():
    strat = {f"C{i:02d}": [("HARD", 26), ("MEDIUM", 27)] for i in range(8)}
    strat["C00"] = [("SOFT", 26), ("MEDIUM", 27)]      # mescola piu veloce
    cond = FieldConditions(53, sc_prob=0, dnf_prob=0)
    df = simulate_field(strat, TYRE, cond, n=1500).set_index("auto")
    assert df.loc["C00", "P_vittoria_%"] >= df["P_vittoria_%"].drop("C00").max()
