"""CLI: scarica una sessione reale via FastF1 e la salva in CSV (una volta sola).

Dopo il download, gli altri script possono lavorare sul CSV anche offline (--csv).
Richiede rete alla prima esecuzione.

Esempio:
    python scripts/fetch_data.py --year 2024 --gp "Monza" --session R
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import data as f1data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--gp", required=True)
    ap.add_argument("--session", default="R")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ses = f1data.load_session(args.year, args.gp, args.session)
    df = f1data.laps_dataframe(ses)
    out = args.out or f"data/{args.year}_{args.gp.replace(' ', '')}_{args.session}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Salvati {len(df)} giri in {out}")
    print("Ora puoi usarlo offline, es.:")
    print(f"  python scripts/train_models.py --csv {out} --tune")
    print(f"  python scripts/optimize_strategy.py --csv {out} --laps {int(df['LapNumber'].max())}")


if __name__ == "__main__":
    main()
