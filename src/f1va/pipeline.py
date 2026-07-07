"""Orchestrazione end-to-end: video -> detection -> tracking -> mappa -> velocità."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import Config
from .detection import CarDetector
from .homography import TrackProjector
from .speed import SpeedEstimator
from .tracking import Tracker, bottom_center
from .viz import draw_minimap, draw_tracks


def run(video_path: str, cfg: Config, out_dir: str = "out") -> pd.DataFrame:
    """Esegue la pipeline su un video e restituisce un DataFrame di telemetria stimata."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossibile aprire il video: {video_path}")
    fps = cfg.speed.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = CarDetector(**cfg.detection)
    tracker = Tracker(**{k: v for k, v in cfg.tracking.items() if k != "tracker"})
    projector = TrackProjector(**cfg.homography)
    speedometer = SpeedEstimator(fps=fps, **{k: v for k, v in cfg.speed.items() if k != "fps"})

    writer = None
    if cfg.output.get("save_video", True):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out / "overlay.mp4"), fourcc, fps, (w, h))

    rows: list[dict] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det = detector.detect(frame)
        tracks = tracker.update(det)
        feet_px = bottom_center(tracks.xyxy)
        world_xy = projector.project(feet_px)
        speeds = speedometer.update(tracks.tracker_id, world_xy)

        t = frame_idx / fps
        for tid, xy in zip(tracks.tracker_id, world_xy):
            rows.append({
                "frame": frame_idx, "time_s": t, "track_id": int(tid),
                "world_x": float(xy[0]), "world_y": float(xy[1]),
                "speed_kmh": speeds.get(int(tid), np.nan),
            })

        if writer is not None:
            vis = draw_tracks(frame, tracks, speeds)
            mini = draw_minimap(world_xy, tracks.tracker_id)
            vis[10:10 + mini.shape[0], w - 10 - mini.shape[1]:w - 10] = mini
            writer.write(vis)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)
    if cfg.output.get("save_csv", True):
        df.to_csv(out / "telemetry.csv", index=False)
    return df
