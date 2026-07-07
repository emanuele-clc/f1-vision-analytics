"""Modelli di machine learning per il tempo sul giro."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class LapTimeModel:
    """Regressione del tempo sul giro (gradient boosting).

    Usa HistGradientBoostingRegressor di scikit-learn: gestisce bene feature miste
    e non-linearita del degrado gomma senza scaling.
    """

    def __init__(self, **kwargs):
        from sklearn.ensemble import HistGradientBoostingRegressor

        params = dict(max_iter=300, learning_rate=0.05, max_depth=4)
        params.update(kwargs)
        self.model = HistGradientBoostingRegressor(**params)
        self.feature_names: list[str] | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "LapTimeModel":
        self.feature_names = list(x.columns)
        self.model.fit(x.to_numpy(), np.asarray(y))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x[self.feature_names].to_numpy())

    def evaluate(self, x: pd.DataFrame, y: pd.Series) -> dict:
        from sklearn.metrics import mean_absolute_error, r2_score

        pred = self.predict(x)
        return {
            "mae_s": round(float(mean_absolute_error(y, pred)), 3),
            "r2": round(float(r2_score(y, pred)), 3),
            "n": int(len(y)),
        }

    def save(self, path: str) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, path)

    @classmethod
    def load(cls, path: str) -> "LapTimeModel":
        import joblib

        blob = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = blob["model"]
        obj.feature_names = blob["features"]
        return obj


def cross_validate(x: pd.DataFrame, y: pd.Series, folds: int = 5) -> dict:
    """K-fold CV del modello tempi sul giro; ritorna MAE medio e deviazione."""
    from sklearn.model_selection import cross_val_score

    m = LapTimeModel()
    m.feature_names = list(x.columns)
    scores = -cross_val_score(
        m.model, x.to_numpy(), np.asarray(y),
        cv=folds, scoring="neg_mean_absolute_error",
    )
    return {"mae_mean_s": round(float(scores.mean()), 3),
            "mae_std_s": round(float(scores.std()), 3), "folds": folds}
