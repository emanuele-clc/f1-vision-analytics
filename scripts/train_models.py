"""CLI: allena il modello di previsione del tempo sul giro su una sessione reale.

Uso:
    python scripts/train_models.py --year 2024 --gp "Monza"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import data as f1data  # noqa: E402
from f1va import features  # noqa: E402
from f1va.models import LapTimeModel, cross_validate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--gp", required=True)
    ap.add_argument("--session", default="R")
    ap.add_argument("--out", default="models/laptime_model.joblib")
    args = ap.parse_args()

    ses = f1data.load_session(args.year, args.gp, args.session)
    laps = f1data.quicklaps(f1data.laps_dataframe(ses))
    x, y = features.build_laptime_dataset(laps)
    print(f"Dataset: {len(y)} giri, {x.shape[1]} feature")
    print("Cross-validation:", cross_validate(x, y))

    model = LapTimeModel().fit(x, y)
    print("Train fit:", model.evaluate(x, y))
    model.save(args.out)
    print(f"Modello salvato in {args.out}")


if __name__ == "__main__":
    main()
