"""Einfache Referenzprognosen. Eingabe: Matrix [Serien x Tage] der Trainingsdaten."""
from __future__ import annotations

import numpy as np


def seasonal_naive(train: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    last_block = train[:, -season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_block, (1, reps))[:, :horizon]


def moving_average(train: np.ndarray, horizon: int, window: int = 28) -> np.ndarray:
    mean = train[:, -window:].mean(axis=1, keepdims=True)
    return np.repeat(mean, horizon, axis=1)
