"""Quantil-Modelle mit LightGBM."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor


def fit_quantile_models(X, y, quantiles, params=None) -> dict:
    """Ein LightGBM-Modell je Quantil."""
    params = params or {}
    return {q: LGBMRegressor(objective="quantile", alpha=q, verbose=-1, random_state=0,
                             **params).fit(X, y)
            for q in quantiles}


def predict_quantiles(models: dict, X) -> dict:
    """Sagt alle Quantile voraus; verhindert sich kreuzende Quantile und negative Werte."""
    qs = sorted(models)
    stacked = np.column_stack([models[q].predict(X) for q in qs])
    stacked = np.maximum.accumulate(np.clip(stacked, 0, None), axis=1)
    return {q: stacked[:, i] for i, q in enumerate(qs)}


def feature_importance(model) -> np.ndarray:
    """Normierte Merkmalswichtigkeit (Summe 1) nach Gain.

    Gain ist der summierte Gewinn aller Splits eines Merkmals. Das ist kein kausaler Effekt;
    korrelierte Merkmale teilen sich die Wichtigkeit.
    """
    imp = np.asarray(model.booster_.feature_importance(importance_type="gain"), float)
    total = imp.sum()
    return imp / total if total > 0 else imp


def feature_importance_all(models: dict) -> dict:
    """Wichtigkeit je Quantilmodell: {q: Array ueber die Merkmale}."""
    return {q: feature_importance(m) for q, m in models.items()}
