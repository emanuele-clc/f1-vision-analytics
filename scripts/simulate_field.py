"""CLI: simulazione a campo pieno (tutte le auto) con posizione in pista e imprevisti.

Esempi:
    python scripts/simulate_field.py --synthetic
    python scripts/simulate_field.py --csv data/2024_Monza_R.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import backtest, features, strategy  # noqa: E402
from f1va.field import FieldConditions, simulate_field  # noqa: E402


def build_from_real(laps, total_laps):
    """Strategie reali per pilota + ordine di partenza (proxy: posizione al giro 1)."""
    tyre = strategy.fit_tyre_models(features.fuel_corrected_degradation(laps))
    strategies, order = {}, []
    for driver in laps["Driver"].unique():
        st = backtest.driver_stints(laps, driver)
        if sum(n for _, n in st) < total_laps * 0.8:
            continue
        if any(c not in tyre for c in {c for c, _ in st}) or len({c for c, _ in st}) < 2:
            continue
        strategies[driver] = st
    lap1 = laps[laps["LapNumber"] == 1].sort_values("Position")
    order = [d for d in lap1["Driver"] if d in strategies]
    order += [d for d in strategies if d not in order]
    return tyre, strategies, order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--laps", type=int, default=53)
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    if args.csv:
        from f1va import data as f1data
        laps = f1data.load_laps_csv(args.csv)
        total = int(laps["LapNumber"].max())
        tyre, strategies, grid = build_from_real(laps, total)
    else:
        from f1va import synthetic
        laps = synthetic.generate_race(n_drivers=20, laps=args.laps, seed=0)
        total = args.laps
        tyre = strategy.fit_tyre_models(features.fuel_corrected_degradation(laps))
        keys = list(tyre)
        rp = total // 2
        plans = [[(keys[0], rp), (keys[1], total - rp)],
                 [(keys[1], rp + 4), (keys[0], total - rp - 4)],
                 [(keys[0], rp // 2), (keys[1], rp), (keys[0], total - rp - rp // 2)]]
        strategies = {f"C{i:02d}": plans[i % len(plans)] for i in range(20)}
        grid = list(strategies)

    if len(strategies) < 2:
        print("Dati insufficienti.")
        return
    cond = FieldConditions(total_laps=total)
    df = simulate_field(strategies, tyre, cond, grid=grid, n=args.n)
    print(f"Simulazione a campo pieno · {len(strategies)} auto · {args.n} gare Monte Carlo\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
