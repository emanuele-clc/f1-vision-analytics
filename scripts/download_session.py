"""CLI: scarica una sessione F1 reale via FastF1 e ne stampa un riassunto.

Uso:
    python scripts/download_session.py --year 2024 --gp "Monza" --session R
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va.strategy import load_session, stint_summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Scarica sessione FastF1")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--gp", required=True, help='es. "Monza"')
    ap.add_argument("--session", default="R", help="FP1/FP2/FP3/Q/R")
    ap.add_argument("--cache", default=".fastf1_cache")
    args = ap.parse_args()

    ses = load_session(args.year, args.gp, args.session, cache=args.cache)
    print(f"Caricata: {args.year} {args.gp} {args.session}")
    print(stint_summary(ses).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
