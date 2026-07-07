"""CLI: prevede la posizione finale e il piazzamento a punti (offline).

Feature note prima della gara: griglia di partenza e forma stagionale di pilota e team.
Validazione realistica: split temporale (allena sulle prime gare, prevede le ultime) e
ranking dei piloti dentro ogni gara (come fa la griglia).

Esempio:
    python scripts/predict_outcome.py --races 45
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import outcome  # noqa: E402


def _rank(scores):
    return np.argsort(np.argsort(scores)) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", type=int, default=45)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="reports")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

    res = outcome.generate_race_outcomes(n_races=args.races, n_drivers=20, seed=args.seed)
    x, y, df = outcome.build_outcome_dataset(res)
    races = sorted(df["Race"].unique())
    cut = int(len(races) * 0.75)
    tr = df.index[df["Race"].isin(races[:cut])]
    te_races = races[cut:]
    print(f"Dataset: {len(y)} risultati · alleno su {cut} gare, prevedo le ultime {len(te_races)}\n")

    # --- posizione finale: predici uno score e ORDINA dentro ogni gara ---
    reg = RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=0)
    reg.fit(x.loc[tr], y.loc[tr])
    pred_pos, true_pos, grid_pos = [], [], []
    for rc in te_races:
        idx = df.index[df["Race"] == rc]
        pred_pos += list(_rank(reg.predict(x.loc[idx])))
        true_pos += list(df.loc[idx, "finish_pos"])
        grid_pos += list(df.loc[idx, "grid_pos"])
    true_pos = np.array(true_pos)
    base = mean_absolute_error(true_pos, grid_pos)      # baseline: arrivi come parti
    mae = mean_absolute_error(true_pos, pred_pos)
    print("== Posizione finale (ranking per gara) ==")
    print(f"Baseline (griglia)  MAE: {base:.2f} posizioni")
    print(f"Modello ML          MAE: {mae:.2f} posizioni  ({(1 - mae/base)*100:+.0f}%)\n")

    # --- zona punti (top 10): classificazione ---
    te = df.index[df["Race"].isin(te_races)]
    clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0)
    clf.fit(x.loc[tr], (y.loc[tr] <= 10).astype(int))
    proba = clf.predict_proba(x.loc[te])[:, 1]
    yte_p = (y.loc[te] <= 10).astype(int)
    acc = accuracy_score(yte_p, (proba >= 0.5).astype(int))
    auc = roc_auc_score(yte_p, proba)
    print("== Piazzamento a punti / top 10 (classificazione) ==")
    print(f"Accuratezza: {acc*100:.1f}%  ·  ROC-AUC: {auc:.3f}\n")

    imp = sorted(zip(x.columns, reg.feature_importances_), key=lambda t: -t[1])
    print("== Importanza feature ==")
    for name, v in imp:
        print(f"  {name:14s} {v:.3f}")

    rep = Path(args.report)
    rep.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    jit = np.random.default_rng(0).normal(0, 0.12, len(true_pos))
    a1.scatter(true_pos + jit, np.array(pred_pos) + jit, s=16, alpha=.45, color="#E10600")
    a1.plot([1, 20], [1, 20], "--", color="#888")
    a1.set_xlabel("Posizione reale")
    a1.set_ylabel("Posizione prevista")
    a1.set_title(f"Posizione finale · MAE {mae:.2f}")
    names = [n for n, _ in imp][::-1]
    vals = [v for _, v in imp][::-1]
    a2.barh(names, vals, color="#FFD12E")
    a2.set_title("Importanza feature")
    fig.tight_layout()
    out = rep / "outcome_model_eval.png"
    fig.savefig(out, dpi=110, facecolor="#0E0E10")
    print(f"\nGrafico salvato in {out}")


if __name__ == "__main__":
    main()
