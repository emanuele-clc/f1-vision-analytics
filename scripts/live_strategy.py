"""CLI: strategia live in-gara (offline).

Mostra la raccomandazione (restare fuori / box) mentre la gara avanza, ricalcolata a
ogni giro dallo stato corrente. Usa il degrado stimato dai dati (sintetici o FastF1).

Esempi:
    python scripts/live_strategy.py --synthetic --laps 53 --start SOFT
    python scripts/live_strategy.py --lap 20 --age 18 --compound MEDIUM --used SOFT,MEDIUM
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import features, strategy  # noqa: E402
from f1va.live import RaceState, recommend  # noqa: E402


def get_tyre_models(args):
    if args.csv:
        from f1va import data as f1data
        laps = f1data.quicklaps(f1data.load_laps_csv(args.csv))
        return strategy.fit_tyre_models(features.fuel_corrected_degradation(laps))
    if args.synthetic or (args.year is None):
        from f1va import synthetic
        laps = synthetic.generate_race(n_drivers=20, laps=args.laps, seed=0)
    else:
        from f1va import data as f1data
        ses = f1data.load_session(args.year, args.gp, "R")
        laps = f1data.quicklaps(f1data.laps_dataframe(ses))
    return strategy.fit_tyre_models(features.fuel_corrected_degradation(laps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", default=None, help="usa un CSV reale (da fetch_data.py)")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--gp", default="Monza")
    ap.add_argument("--laps", type=int, default=53)
    ap.add_argument("--start", default="SOFT", help="mescola di partenza (demo)")
    ap.add_argument("--pit-loss", type=float, default=22.0)
    # stato singolo (opzionale)
    ap.add_argument("--lap", type=int)
    ap.add_argument("--age", type=int)
    ap.add_argument("--compound")
    ap.add_argument("--used", default="")
    args = ap.parse_args()

    tyre = get_tyre_models(args)
    comp = list(tyre)
    if len(comp) < 2:
        print("Servono almeno due mescole con dati sufficienti.")
        return

    if args.lap is not None:                      # stato singolo
        used = frozenset(c for c in args.used.upper().split(",") if c) | {args.compound}
        state = RaceState(args.laps, args.lap, args.compound, args.age, used, args.pit_loss)
        print(recommend(state, tyre, comp).action)
        return

    # demo: percorre la gara e stampa i cambi di raccomandazione
    print(f"Gara di {args.laps} giri · partenza su {args.start}\n")
    start = args.start if args.start in tyre else comp[0]
    compound, age, used, last_key = start, 0, {start}, None
    for lap in range(1, args.laps):
        state = RaceState(args.laps, lap, compound, age, frozenset(used), args.pit_loss)
        plan = recommend(state, tyre, comp)
        # stampa solo quando cambia la decisione (non il conto alla rovescia)
        target = plan.next_stints[0][0] if plan.next_stints else "FINISH"
        pit_lap = lap + plan.stay_out_laps if plan.next_stints else None
        key = (target, pit_lap)
        if key != last_key:
            print(f"Giro {lap:>2}: {plan.action}")
            last_key = key
        if plan.stay_out_laps <= 1 and plan.next_stints:
            compound = plan.next_stints[0][0]
            used.add(compound)
            age = 0
            print(f"        → SOSTA: monta {compound} (giro {lap})")
            last_key = None
        else:
            age += 1


if __name__ == "__main__":
    main()
