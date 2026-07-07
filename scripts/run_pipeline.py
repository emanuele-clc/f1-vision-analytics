"""CLI: esegue la pipeline video -> telemetria stimata.

Uso:
    python scripts/run_pipeline.py --video data/samples/onboard.mp4 --out out/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1va.config import load_config  # noqa: E402
from f1va.pipeline import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="F1 Vision Analytics — pipeline")
    ap.add_argument("--video", required=True, help="percorso del video di input")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = run(args.video, cfg, out_dir=args.out)
    print(f"Fatto. {len(df)} righe di telemetria stimata -> {args.out}/telemetry.csv")


if __name__ == "__main__":
    main()
