"""Test della strategia live in-gara."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va.live import RaceState, recommend  # noqa: E402
from f1va.strategy import TyreModel  # noqa: E402

TYRE = {"SOFT": TyreModel(78.0, 0.14), "MEDIUM": TyreModel(78.6, 0.09),
        "HARD": TyreModel(79.3, 0.05)}


def test_remaining_laps_add_up():
    st = RaceState(53, 12, "SOFT", 12, frozenset({"SOFT"}))
    p = recommend(st, TYRE, list(TYRE), max_further_stops=2)
    covered = p.stay_out_laps + sum(n for _, n in p.next_stints)
    assert covered == 53 - 12


def test_two_compound_rule_forces_a_stop():
    # una sola mescola usata finora -> deve prevedere una sosta su mescola diversa
    st = RaceState(53, 10, "SOFT", 10, frozenset({"SOFT"}))
    p = recommend(st, TYRE, list(TYRE), max_further_stops=2)
    assert p.next_stints
    assert p.next_stints[0][0] != "SOFT"


def test_near_end_stays_out():
    # a fine gara, con due mescole gia usate, conviene finire
    st = RaceState(53, 49, "HARD", 8, frozenset({"SOFT", "HARD"}))
    p = recommend(st, TYRE, list(TYRE), max_further_stops=2)
    assert p.stay_out_laps == 53 - 49
    assert p.next_stints == ()
