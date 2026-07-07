"""CLI: genera una GIF del replay delle posizioni da una sessione reale (FastF1).

Uso:
    python scripts/make_replay_gif.py --year 2024 --gp "Monza" --seconds 90 --out images/replay.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va import data as f1data  # noqa: E402
from f1va import replay as rp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--gp", required=True)
    ap.add_argument("--session", default="R")
    ap.add_argument("--seconds", type=int, default=90, help="durata da inizio sessione")
    ap.add_argument("--step", type=float, default=1.5, help="risoluzione temporale (s)")
    ap.add_argument("--out", default="images/replay.gif")
    args = ap.parse_args()

    ses = f1data.load_session(args.year, args.gp, args.session)
    outline = rp.track_outline(ses)
    frames = [f for f in rp.build_frames(ses, step_s=args.step) if f["t"] <= args.seconds]
    drivers = sorted({d for f in frames for d in f["cars"]})
    print(f"{len(frames)} frame · {len(drivers)} piloti")

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=110)
    fig.patch.set_facecolor("#0E0E10")
    ax.set_facecolor("#0E0E10")
    ax.plot(outline["x"], outline["y"], color="#2A2A33", lw=10, solid_capstyle="round")
    dots = ax.scatter([], [], s=80, c="#E10600", edgecolors="#0E0E10", linewidths=1, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{args.year} {args.gp} — Replay", color="#F5F5F5",
                 fontsize=12, fontweight="bold")

    def update(i):
        fr = frames[i]
        pts = [(fr["cars"][d]["x"], fr["cars"][d]["y"]) for d in drivers if d in fr["cars"]]
        dots.set_offsets(pts)
        return (dots,)

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=15),
              savefig_kwargs={"facecolor": "#0E0E10"})
    print(f"GIF salvata in {args.out}")


if __name__ == "__main__":
    main()
