"""CLI: duello testa a testa contro un rivale, con aria sporca e imprevisti (offline).

Trova la nostra strategia che massimizza la probabilita di finire davanti al rivale,
considerando Safety Car, ritiri e pioggia.

Esempi:
    python scripts/duel.py --synthetic --laps 53 --gap -0.8
    python scripts/duel.py --csv data/2024_Monza_R.csv --laps 53 --gap 1.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import features, strategy  # noqa: E402
from f1va.duel import DuelConditions, best_response, duel_win_probability  # noqa: E402


def get_tyre(args):
    if args.csv:
        from f1va import data as f1data
        laps = f1data.quicklaps(f1data.load_laps_csv(args.csv))
    else:
        from f1va import synthetic
        laps = synthetic.generate_race(n_drivers=20, laps=args.laps, seed=0)
    return strategy.fit_tyre_models(features.fuel_corrected_degradation(laps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--laps", type=int, default=53)
    ap.add_argument("--gap", type=float, default=-0.8,
                    help="distacco iniziale dal rivale (s); negativo = siamo dietro")
    args = ap.parse_args()

    tyre = get_tyre(args)
    comp = list(tyre)
    if len(comp) < 2:
        print("Servono almeno due mescole.")
        return

    # strategia ipotizzata del rivale: 1 sosta bilanciata
    rival = [(comp[0], args.laps // 2), (comp[1], args.laps - args.laps // 2)]
    cond = DuelConditions(total_laps=args.laps)

    br = best_response(rival, tyre, cond, comp, start_gap=args.gap)
    print(f"Rivale: {' -> '.join(f'{c}({n})' for c,n in rival)} (pit al giro {br['rival_first_pit']})")
    print(f"Distacco iniziale: {args.gap:+.1f}s\n")
    print("Nostre opzioni (probabilita di finire davanti):")
    for c in br["candidates"]:
        plan = " -> ".join(f"{cc}({n})" for cc, n in c["stints"])
        print(f"  {plan:34s}  {c['win_prob']*100:5.1f}%   [{c['label']}]")
    b = br["best"]
    print(f"\nRisposta consigliata: {' -> '.join(f'{c}({n})' for c,n in b['stints'])} "
          f"({b['label']}) → {b['win_prob']*100:.0f}% di finire davanti")

    # impatto degli imprevisti sulla stessa strategia
    print("\nImpatto degli imprevisti (nostra strategia consigliata):")
    for label, kw in [("clean", {"sc_prob": 0, "dnf_prob": 0, "rain_prob": 0}),
                      ("Safety Car", {"sc_prob": 0.8, "dnf_prob": 0, "rain_prob": 0}),
                      ("pioggia", {"sc_prob": 0, "dnf_prob": 0, "rain_prob": 0.8})]:
        cc = DuelConditions(total_laps=args.laps, **kw)
        p = duel_win_probability(b["stints"], rival, tyre, cc, start_gap=args.gap)
        print(f"  {label:12s} {p*100:5.1f}%")


if __name__ == "__main__":
    main()
