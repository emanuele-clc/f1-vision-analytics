"""Test del simulatore Monte Carlo della strategia."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va.montecarlo import (  # noqa: E402
    RaceConditions, optimize_mc, simulate_strategy_mc, win_probability,
)
from f1va.strategy import TyreModel, simulate_strategy  # noqa: E402

TYRE = {"SOFT": TyreModel(78.0, 0.14), "MEDIUM": TyreModel(78.6, 0.09),
        "HARD": TyreModel(79.3, 0.05)}
STINTS = [("SOFT", 20), ("HARD", 33)]


def test_mc_matches_deterministic_without_noise():
    cond = RaceConditions(total_laps=53, pit_loss_sd=0, sc_prob=0, deg_noise_sd=0)
    mc = simulate_strategy_mc(STINTS, TYRE, cond, n=500)
    det = simulate_strategy(53, STINTS, TYRE, pit_loss=cond.pit_loss)
    assert mc["mean_s"] == pytest.approx(det, abs=1e-6)
    assert mc["std_s"] == pytest.approx(0.0, abs=1e-6)


def test_safety_car_reduces_expected_time():
    no_sc = simulate_strategy_mc(STINTS, TYRE, RaceConditions(53, sc_prob=0.0), n=4000)
    with_sc = simulate_strategy_mc(STINTS, TYRE, RaceConditions(53, sc_prob=0.6), n=4000)
    assert with_sc["mean_s"] < no_sc["mean_s"]      # la SC rende il pit meno costoso


def test_win_probability_bounds_and_symmetry():
    cond = RaceConditions(total_laps=53)
    a, b = [("SOFT", 26), ("HARD", 27)], [("MEDIUM", 26), ("HARD", 27)]
    p = win_probability(a, b, TYRE, cond, n=3000)
    assert 0.0 <= p <= 1.0


def test_optimize_returns_valid_plan():
    cond = RaceConditions(total_laps=53, sc_prob=0.4)
    out = optimize_mc(TYRE, cond, list(TYRE), max_stops=2, n=2000)
    assert sum(n for _, n in out["best"]["stints"]) == 53
    assert 0.0 <= out["win_prob_vs_second"] <= 1.0
    assert len({c for c, _ in out["best"]["stints"]}) >= 2
