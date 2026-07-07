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


def compare_models(x: pd.DataFrame, y: pd.Series, cv: int = 5) -> pd.DataFrame:
    """Confronta piu algoritmi con cross-validation (MAE). Baseline inclusa."""
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    candidates = {
        "Baseline (media)": DummyRegressor(strategy="mean"),
        "Regressione lineare": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05),
    }
    rows = []
    xn, yn = x.to_numpy(), np.asarray(y)
    for name, mdl in candidates.items():
        sc = -cross_val_score(mdl, xn, yn, cv=cv, scoring="neg_mean_absolute_error")
        rows.append({"modello": name, "mae_mean_s": round(float(sc.mean()), 3),
                     "mae_std_s": round(float(sc.std()), 3)})
    return pd.DataFrame(rows)


def tune_laptime_model(x: pd.DataFrame, y: pd.Series, n_iter: int = 20,
                       cv: int = 4, seed: int = 0) -> tuple["LapTimeModel", dict]:
    """Ricerca casuale degli iperparametri del gradient boosting (MAE, cross-validation)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import RandomizedSearchCV

    space = {
        "max_iter": [300, 500, 800, 1200],
        "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [3, 4, 6, None],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0],
    }
    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(), space, n_iter=n_iter, cv=cv,
        scoring="neg_mean_absolute_error", random_state=seed, n_jobs=-1)
    search.fit(x.to_numpy(), np.asarray(y))
    m = LapTimeModel.__new__(LapTimeModel)
    m.model = search.best_estimator_
    m.feature_names = list(x.columns)
    return m, {"best_params": search.best_params_, "cv_mae_s": round(-search.best_score_, 3)}


def permutation_importance_table(model: "LapTimeModel", x: pd.DataFrame,
                                 y: pd.Series, n_repeats: int = 6) -> pd.DataFrame:
    """Importanza delle feature per permutazione (quanto peggiora l'errore mescolandole)."""
    from sklearn.inspection import permutation_importance

    r = permutation_importance(model.model, x[model.feature_names].to_numpy(), np.asarray(y),
                               n_repeats=n_repeats, random_state=0,
                               scoring="neg_mean_absolute_error")
    return (pd.DataFrame({"feature": model.feature_names,
                          "importanza_s": r.importances_mean.round(4)})
            .sort_values("importanza_s", ascending=False).reset_index(drop=True))


def stacked_model():
    """Ensemble stacking: GBM + Random Forest, combinati da un meta-modello lineare."""
    from sklearn.ensemble import (
        HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor,
    )
    from sklearn.linear_model import RidgeCV

    estimators = [
        ("gbm", HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05)),
        ("rf", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)),
    ]
    return StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=-1)


def learning_curve_data(x: pd.DataFrame, y: pd.Series, cv: int = 5):
    """Curva di apprendimento: MAE (train/validation) al crescere dei dati."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import learning_curve

    sizes, train, val = learning_curve(
        HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05),
        x.to_numpy(), np.asarray(y), cv=cv, scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)
    return sizes, -train.mean(axis=1), -val.mean(axis=1)
