"""CLI: backtest del motore di strategia su gare reali (o sintetiche).

Confronta la strategia consigliata con quella realmente adottata e quantifica il tempo
lasciato sul tavolo. E la validazione piu concreta di un motore di strategia.

Esempi:
    python scripts/backtest.py --synthetic
    python scripts/backtest.py --csv data/2024_Monza_R.csv
    python scripts/backtest.py --races 2024:Monza 2024:Silverstone 2024:Spa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import backtest  # noqa: E402


def _load_real(spec: str):
    from f1va import data as f1data
    year, gp = spec.split(":", 1)
    ses = f1data.load_session(int(year), gp, "R")
    return f1data.laps_dataframe(ses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--races", nargs="*", default=None, help='es. 2024:Monza 2024:Spa')
    ap.add_argument("--report", default="reports")
    args = ap.parse_args()

    if args.races:                       # backtest multi-gara su dati reali
        sessions = [(s, _load_real(s)) for s in args.races]
        table = backtest.backtest_sessions(sessions)
        print(table.to_string(index=False))
        print(f"\nMatch soste medio: {table['match_soste_%'].mean():.0f}%  ·  "
              f"delta mediano medio: {table['delta_mediano_s'].mean():.1f}s")
        return

    if args.csv:
        from f1va import data as f1data
        laps = f1data.load_laps_csv(args.csv)
        label = Path(args.csv).stem
    else:
        from f1va import synthetic
        laps = synthetic.generate_race(n_drivers=20, laps=53, seed=0)
        label = "gara sintetica"

    res = backtest.backtest_race(laps)
    rec = res["recommended"]
    df = res["drivers"]
    m = res["metrics"]
    print(f"Backtest: {label}")
    print(f"Strategia consigliata dal motore: {' -> '.join(f'{c}({n})' for c,n in rec.stints)}\n")
    print(df.to_string(index=False))
    print(f"\nPiloti analizzati: {m['drivers']}")
    print(f"Match sul numero di soste: {m['match_stops_pct']}%")
    print(f"Tempo mediano lasciato sul tavolo vs ottimo: {m['median_delta_s']}s")

    rep = Path(args.report)
    rep.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(df["driver"], df["delta_s"], color="#E10600")
    ax.set_xlabel("Tempo perso vs strategia ottima (s)")
    ax.set_title(f"Backtest strategia · {label}")
    ax.invert_yaxis()
    fig.tight_layout()
    out = rep / "backtest.png"
    fig.savefig(out, dpi=110, facecolor="#0E0E10")
    print(f"Grafico salvato in {out}")


if __name__ == "__main__":
    main()
