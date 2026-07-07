"""CLI: allena e valida in modo rigoroso il modello ML dei tempi sul giro.

Include: confronto tra piu algoritmi, ricerca degli iperparametri, importanza delle
feature e grafici. Funziona anche offline con dati sintetici realistici.

Esempi:
    python scripts/train_models.py --synthetic --tune          # completo, offline
    python scripts/train_models.py --synthetic --races 12
    python scripts/train_models.py --year 2024 --gp "Monza"    # dati reali FastF1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import features, models  # noqa: E402


def get_laps(args):
    if args.csv:
        from f1va import data as f1data
        print(f"Dati: CSV {args.csv}")
        return f1data.quicklaps(f1data.load_laps_csv(args.csv))
    if args.synthetic:
        from f1va import synthetic
        print(f"Dati: stagione sintetica ({args.races} gare, offline)")
        races = max(args.races, 30) if args.heavy else args.races
        return synthetic.generate_season(n_races=races, n_drivers=20, seed=args.seed)
    from f1va import data as f1data
    print(f"Dati: FastF1 {args.year} {args.gp} {args.session}")
    ses = f1data.load_session(args.year, args.gp, args.session)
    return f1data.quicklaps(f1data.laps_dataframe(ses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", default=None, help="usa un CSV reale (da fetch_data.py)")
    ap.add_argument("--tune", action="store_true", help="ricerca iperparametri (piu lenta)")
    ap.add_argument("--heavy", action="store_true",
                    help="training rigoroso: dataset ampio, ricerca estesa, curva di apprendimento, ensemble")
    ap.add_argument("--races", type=int, default=10, help="gare sintetiche")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--gp", default="Monza")
    ap.add_argument("--session", default="R")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/laptime_model.joblib")
    ap.add_argument("--report", default="reports")
    args = ap.parse_args()

    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    laps = get_laps(args)
    x, y = features.build_laptime_dataset(laps)
    print(f"Dataset: {len(y):,} giri · {x.shape[1]} feature\n")

    print("== Confronto modelli (5-fold CV, MAE in secondi) ==")
    print(models.compare_models(x, y).to_string(index=False), "\n")

    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=args.seed)

    if args.tune or args.heavy:
        n_iter, cv = (60, 5) if args.heavy else (20, 4)
        print(f"== Ricerca iperparametri (RandomizedSearchCV, {n_iter} iter × {cv} fold)… ==")
        model, info = models.tune_laptime_model(xtr, ytr, n_iter=n_iter, cv=cv, seed=args.seed)
        print("Migliori parametri:", info["best_params"])
        print("CV MAE:", info["cv_mae_s"], "s\n")
    else:
        model = models.LapTimeModel().fit(xtr, ytr)

    baseline = mean_absolute_error(yte, [ytr.mean()] * len(yte))
    metrics = model.evaluate(xte, yte)
    print("== Valutazione su test set (20%) ==")
    print(f"Baseline (media)  MAE: {baseline:.3f}s")
    print(f"Modello finale    MAE: {metrics['mae_s']}s · R2: {metrics['r2']}")
    print(f"Miglioramento: {(1 - metrics['mae_s'] / baseline) * 100:.0f}% sul baseline\n")

    print("== Importanza delle feature (permutazione) ==")
    imp = models.permutation_importance_table(model, xte, yte)
    print(imp.to_string(index=False), "\n")

    # grafici (3 pannelli)
    rep = Path(args.report)
    rep.mkdir(parents=True, exist_ok=True)
    pred = model.predict(xte)
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))
    ax1.scatter(yte, pred, s=8, alpha=.4, color="#E10600")
    lims = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
    ax1.plot(lims, lims, "--", color="#888")
    ax1.set_xlabel("Reale (s)")
    ax1.set_ylabel("Predetto (s)")
    ax1.set_title(f"Predetto vs reale · MAE {metrics['mae_s']}s")
    ax2.hist(pred - yte.to_numpy(), bins=40, color="#0090FF", alpha=.8)
    ax2.set_xlabel("Errore (s)")
    ax2.set_ylabel("Giri")
    ax2.set_title("Distribuzione degli errori")
    top = imp.head(6).iloc[::-1]
    ax3.barh(top["feature"], top["importanza_s"], color="#FFD12E")
    ax3.set_xlabel("Importanza (s)")
    ax3.set_title("Feature piu importanti")
    fig.tight_layout()
    plot_path = rep / "laptime_model_eval.png"
    fig.savefig(plot_path, dpi=110, facecolor="#0E0E10")
    print(f"Grafico salvato in {plot_path}")

    if args.heavy:
        from sklearn.metrics import mean_absolute_error as _mae
        print("\n== Ensemble stacking (GBM + Random Forest) ==")
        stack = models.stacked_model().fit(xtr, ytr)
        print(f"Stacking MAE: {_mae(yte, stack.predict(xte)):.3f}s")
        print("\n== Curva di apprendimento ==")
        sizes, tr_mae, val_mae = models.learning_curve_data(x, y, cv=5)
        for sz, tm, vm in zip(sizes, tr_mae, val_mae):
            print(f"  {int(sz):>6} giri  train {tm:.3f}s  validation {vm:.3f}s")
        figl, axl = plt.subplots(figsize=(6.5, 4.2))
        axl.plot(sizes, tr_mae, "-o", color="#E10600", label="train")
        axl.plot(sizes, val_mae, "-o", color="#0090FF", label="validation")
        axl.set_xlabel("Numero di giri di training")
        axl.set_ylabel("MAE (s)")
        axl.set_title("Curva di apprendimento")
        axl.legend()
        figl.tight_layout()
        figl.savefig(rep / "learning_curve.png", dpi=110, facecolor="#0E0E10")
        print(f"Curva salvata in {rep / 'learning_curve.png'}")

    model.save(args.out)
    print(f"Modello salvato in {args.out}")
    print(f"\nTempo totale: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
