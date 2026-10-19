"""Quantil-Modelle (LightGBM oder scikit-learn)."""
from __future__ import annotations

import numpy as np


def _make_model(q: float, backend: str, params: dict):
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(objective="quantile", alpha=q, verbose=-1, random_state=0, **params)
    if backend == "sklearn":
        from sklearn.ensemble import HistGradientBoostingRegressor
        p = {"max_iter": params.get("n_estimators", 200),
             "learning_rate": params.get("learning_rate", 0.05)}
        return HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=0, **p)
    raise ValueError(f"Unbekanntes Backend: {backend}")


def fit_quantile_models(X, y, quantiles, backend="lightgbm", params=None) -> dict:
    params = params or {}
    return {q: _make_model(q, backend, params).fit(X, y) for q in quantiles}


def predict_quantiles(models: dict, X) -> dict:
    """Sagt alle Quantile voraus; verhindert sich kreuzende Quantile und negative Werte."""
    qs = sorted(models)
    stacked = np.column_stack([models[q].predict(X) for q in qs])
    stacked = np.maximum.accumulate(np.clip(stacked, 0, None), axis=1)
    return {q: stacked[:, i] for i, q in enumerate(qs)}
