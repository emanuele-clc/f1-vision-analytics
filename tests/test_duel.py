"""Test del duello testa a testa e degli imprevisti."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va.duel import DuelConditions, duel_win_probability  # noqa: E402
from f1va.strategy import TyreModel  # noqa: E402

TYRE = {"SOFT": TyreModel(78.0, 0.14), "MEDIUM": TyreModel(78.6, 0.09),
        "HARD": TyreModel(79.3, 0.05)}
PLAN = [("SOFT", 20), ("HARD", 33)]


def test_leader_certain_without_noise_or_events():
    cond = DuelConditions(53, lap_noise_sd=0, pit_loss_sd=0, sc_prob=0, dnf_prob=0, rain_prob=0)
    assert duel_win_probability(PLAN, PLAN, TYRE, cond, start_gap=5, n=400) == 1.0


def test_win_probability_bounds():
    cond = DuelConditions(53)
    p = duel_win_probability(PLAN, PLAN, TYRE, cond, start_gap=0, n=1000)
    assert 0.0 <= p <= 1.0


def test_safety_car_reduces_leader_certainty():
    base = DuelConditions(53, sc_prob=0, dnf_prob=0, rain_prob=0)
    sc = DuelConditions(53, sc_prob=0.9, dnf_prob=0, rain_prob=0)
    p_base = duel_win_probability(PLAN, PLAN, TYRE, base, start_gap=3, n=3000)
    p_sc = duel_win_probability(PLAN, PLAN, TYRE, sc, start_gap=3, n=3000)
    assert p_sc < p_base


def test_rain_adds_upset_chance():
    dry = DuelConditions(53, sc_prob=0, dnf_prob=0, rain_prob=0)
    wet = DuelConditions(53, sc_prob=0, dnf_prob=0, rain_prob=0.9)
    p_dry = duel_win_probability(PLAN, PLAN, TYRE, dry, start_gap=3, n=3000)
    p_wet = duel_win_probability(PLAN, PLAN, TYRE, wet, start_gap=3, n=3000)
    assert p_wet < p_dry
