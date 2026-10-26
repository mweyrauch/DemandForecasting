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


def feature_importance(model, feats, backend, X=None, y=None, q: float = 0.5,
                       n_sample: int = 3000, seed: int = 0) -> np.ndarray:
    """Normierte Merkmalswichtigkeit (Summe 1).

    lightgbm: Gain, also der summierte Gewinn aller Splits eines Merkmals.
    sklearn:  Permutationswichtigkeit auf einer Stichprobe (Anstieg des Pinball-Loss).
    Hinweis: Beide Masse sind keine kausalen Effekte; korrelierte Merkmale teilen sich die
    Wichtigkeit.
    """
    if backend == "lightgbm":
        imp = np.asarray(model.booster_.feature_importance(importance_type="gain"), float)
    elif backend == "sklearn":
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import make_scorer, mean_pinball_loss
        idx = np.random.default_rng(seed).choice(len(X), size=min(n_sample, len(X)), replace=False)
        scorer = make_scorer(mean_pinball_loss, alpha=q, greater_is_better=False)
        res = permutation_importance(model, X.iloc[idx][feats], np.asarray(y)[idx], scoring=scorer,
                                     n_repeats=2, random_state=seed)
        imp = np.clip(res.importances_mean, 0, None)
    else:
        raise ValueError(f"Unbekanntes Backend: {backend}")
    total = imp.sum()
    return imp / total if total > 0 else imp


def feature_importance_all(models: dict, feats, backend, X=None, y=None) -> dict:
    """Wichtigkeit je Quantilmodell: {q: Array ueber die Merkmale}."""
    return {q: feature_importance(m, feats, backend, X, y, q) for q, m in models.items()}
