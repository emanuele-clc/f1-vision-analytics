"""CLI: simulazione Monte Carlo della strategia sotto incertezza (offline).

Stima il degrado dai dati (reali FastF1 o sintetici) e valuta le strategie considerando
Safety Car, variabilita del degrado e del pit-stop.

Esempi:
    python scripts/simulate_race.py --synthetic --laps 53 --sc-prob 0.4
    python scripts/simulate_race.py --year 2024 --gp "Monza" --laps 53
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import features, strategy  # noqa: E402
from f1va.montecarlo import RaceConditions, optimize_mc  # noqa: E402


def get_deg(args):
    if args.csv:
        from f1va import data as f1data
        return features.fuel_corrected_degradation(f1data.quicklaps(f1data.load_laps_csv(args.csv)))
    if args.synthetic:
        from f1va import synthetic
        laps = synthetic.generate_race(n_drivers=20, laps=args.laps, seed=0)
        return features.fuel_corrected_degradation(laps)
    from f1va import data as f1data
    ses = f1data.load_session(args.year, args.gp, "R")
    laps = f1data.quicklaps(f1data.laps_dataframe(ses))
    return features.fuel_corrected_degradation(laps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", default=None, help="usa un CSV reale (da fetch_data.py)")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--gp", default="Monza")
    ap.add_argument("--laps", type=int, default=53)
    ap.add_argument("--sc-prob", type=float, default=0.35)
    ap.add_argument("--pit-loss", type=float, default=22.0)
    ap.add_argument("--n", type=int, default=8000)
    args = ap.parse_args()

    deg = get_deg(args)
    tyre = strategy.fit_tyre_models(deg)
    if len(tyre) < 2:
        print("Dati insufficienti (servono >=2 mescole).")
        return

    cond = RaceConditions(total_laps=args.laps, pit_loss=args.pit_loss, sc_prob=args.sc_prob)
    out = optimize_mc(tyre, cond, list(tyre), max_stops=2, n=args.n)

    print(f"Strategie valutate: {out['n_evaluated']}  ·  Safety Car prob: {args.sc_prob:.0%}\n")
    print("Top strategie (tempo atteso):")
    for r in out["ranked"][:4]:
        plan = " -> ".join(f"{c}({n})" for c, n in r["stints"])
        print(f"  {plan:34s}  {r['mean_s']/60:6.2f} min   "
              f"(P10-P90 {r['p10_s']/60:.2f}-{r['p90_s']/60:.2f})")
    b = out["best"]
    print(f"\nConsigliata: {' -> '.join(f'{c}({n})' for c, n in b['stints'])}")
    print(f"Batte la seconda strategia nel {out['win_prob_vs_second']*100:.0f}% delle simulazioni")


if __name__ == "__main__":
    main()
