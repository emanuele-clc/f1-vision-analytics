"""CLI: stima il degrado dai dati reali e calcola la strategia ottimale.

Uso:
    python scripts/optimize_strategy.py --year 2024 --gp "Monza" --laps 53
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import data as f1data  # noqa: E402
from f1va import features, strategy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--gp", required=True)
    ap.add_argument("--laps", type=int, required=True)
    ap.add_argument("--max-stops", type=int, default=2)
    ap.add_argument("--fuel-correct", action="store_true",
                    help="separa il degrado gomma dall'effetto carburante")
    args = ap.parse_args()

    ses = f1data.load_session(args.year, args.gp, "R")
    laps = f1data.quicklaps(f1data.laps_dataframe(ses))
    deg = (features.fuel_corrected_degradation(laps) if args.fuel_correct
           else features.degradation_table(laps))
    print("Degrado stimato:\n", deg.to_string(index=False))

    models = strategy.fit_tyre_models(deg)
    best = strategy.optimize_strategy(args.laps, list(models.keys()), models,
                                      max_stops=args.max_stops)
    plan = " -> ".join(f"{c}({n})" for c, n in best.stints)
    print(f"\nStrategia ottimale: {plan}")
    print(f"Soste: {best.n_stops} | Tempo gara stimato: {best.total_time_s/60:.2f} min")


if __name__ == "__main__":
    main()
