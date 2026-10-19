"""Metriken und zeitliche Validierung."""
from __future__ import annotations

import numpy as np


def wape(y, f) -> float:
    """Weighted Absolute Percentage Error: sum|y-f| / sum|y|."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - f).sum() / denom) if denom > 0 else float("nan")


def mase(y, f, y_train, season: int = 1) -> float:
    """Mean Absolute Scaled Error ueber alle Serien (Matrizen: Serien x Tage)."""
    y, f, y_train = (np.atleast_2d(np.asarray(a, float)) for a in (y, f, y_train))
    scale = np.abs(y_train[:, season:] - y_train[:, :-season]).mean(axis=1)
    err = np.abs(y - f).mean(axis=1)
    ok = scale > 0
    return float((err[ok] / scale[ok]).mean()) if ok.any() else float("nan")


def pinball_loss(y, pred, q: float) -> float:
    """Pinball-Loss fuer das Quantil q."""
    d = np.asarray(y, float) - np.asarray(pred, float)
    return float(np.maximum(q * d, (q - 1) * d).mean())


def calibration(y, pred) -> float:
    """Anteil der Beobachtungen <= Quantilsprognose. Sollte ungefaehr q entsprechen."""
    return float((np.asarray(y, float) <= np.asarray(pred, float)).mean())


def rolling_origin_cutoffs(n_days: int, horizon: int, n_splits: int, step: int | None = None):
    """Trainingsende (exklusiv) fuer rollierende Zeitreihen-Validierung.

    Jeder Split trainiert auf ``[:cutoff]`` und testet auf ``[cutoff:cutoff+horizon]``.
    """
    step = step or horizon
    last = n_days - horizon
    cutoffs = [last - i * step for i in range(n_splits)]
    return sorted(c for c in cutoffs if c > 0)
