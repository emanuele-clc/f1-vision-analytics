"""Test del backtester (dati sintetici, offline)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import backtest, synthetic  # noqa: E402


def test_driver_stints_reconstructed():
    laps = synthetic.generate_race(n_drivers=6, laps=40, seed=1)
    d = laps["Driver"].iloc[0]
    stints = backtest.driver_stints(laps, d)
    assert sum(n for _, n in stints) == laps[laps["Driver"] == d]["LapNumber"].nunique()
    assert all(isinstance(c, str) and n > 0 for c, n in stints)


def test_backtest_race_produces_metrics():
    laps = synthetic.generate_race(n_drivers=20, laps=53, seed=2)
    res = backtest.backtest_race(laps, total_laps=53)
    assert res["recommended"] is not None
    assert not res["drivers"].empty
    m = res["metrics"]
    assert 0 <= m["match_stops_pct"] <= 100
    # il tempo reale non puo essere migliore dell'ottimo del motore (delta >= ~0)
    assert res["drivers"]["delta_s"].min() >= -1.0


def test_backtest_sessions_aggregates():
    sessions = [("A", synthetic.generate_race(20, 53, 3)),
                ("B", synthetic.generate_race(20, 60, 4))]
    out = backtest.backtest_sessions(sessions)
    assert len(out) == 2
    assert {"gara", "consigliata", "match_soste_%"} <= set(out.columns)
