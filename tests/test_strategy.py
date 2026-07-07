"""Test del simulatore e dell'ottimizzatore di strategia (logica pura)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va.strategy import (  # noqa: E402
    TyreModel, simulate_strategy, optimize_strategy, _compositions,
)


def test_stint_time():
    # base 90, deg 0.5: giri con eta 0,1,2 -> 90 + 90.5 + 91 = 271.5
    tm = TyreModel(base=90.0, deg=0.5)
    assert tm.stint_time(3) == pytest.approx(271.5)


def test_simulate_requires_matching_laps():
    models = {"SOFT": TyreModel(90, 0.5)}
    with pytest.raises(ValueError):
        simulate_strategy(10, [("SOFT", 8)], models)


def test_simulate_adds_pit_loss():
    models = {"SOFT": TyreModel(90, 0.0), "HARD": TyreModel(90, 0.0)}
    # 2 stint da 5 giri: 10*90 + 1 sosta*22 = 922
    t = simulate_strategy(10, [("SOFT", 5), ("HARD", 5)], models, pit_loss=22.0)
    assert t == pytest.approx(922.0)


def test_compositions_respect_min():
    parts = list(_compositions(20, 2, 5))
    assert all(sum(p) == 20 and min(p) >= 5 for p in parts)
    assert (5, 15) in parts and (10, 10) in parts


def test_optimizer_prefers_low_degradation():
    # HARD degrada meno: su gara lunga l'ottimo deve usarla molto
    models = {"SOFT": TyreModel(88, 0.9), "HARD": TyreModel(90, 0.1)}
    best = optimize_strategy(40, ["SOFT", "HARD"], models, max_stops=2)
    laps_on_hard = sum(n for c, n in best.stints if c == "HARD")
    assert laps_on_hard >= 20
    assert len({c for c, _ in best.stints}) >= 2  # regola due mescole


def test_optimizer_two_compound_rule():
    models = {"SOFT": TyreModel(88, 0.2), "HARD": TyreModel(90, 0.2)}
    best = optimize_strategy(30, ["SOFT", "HARD"], models, max_stops=2,
                             require_two_compounds=True)
    assert len({c for c, _ in best.stints}) >= 2
