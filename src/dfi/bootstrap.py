"""Bootstrap ueber Zeitreihen (Artikel) fuer Konfidenzintervalle.

Idee: Die Artikel werden mit Zuruecklegen neu gezogen (jede Ziehung als Gewichtsvektor ueber
die Artikel). Alle Kennzahlen sind gewichtete Summen ueber die Artikel und lassen sich daher
mit Matrixprodukten fuer viele Ziehungen gleichzeitig berechnen. Dieselben Ziehungen werden fuer
alle Politiken verwendet (gepaarter Vergleich).

Hinweis: Das Verfahren erfasst die Unsicherheit ueber die Artikelauswahl, nicht ueber die Zeit.
Die rollierenden Cutoffs decken den zeitlichen Aspekt nur grob ab.
"""
from __future__ import annotations

import numpy as np

from .inventory import inventory_at_service_level


def bootstrap_counts(n: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """Gewichte (n_boot x n): wie oft jeder Artikel in einer Ziehung vorkommt."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)


def interval(draws, level: float = 0.95) -> tuple[float, float]:
    x = np.asarray(draws, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(x, [a, 100 - a])
    return float(lo), float(hi)


def curve_arrays(C, lost, held, demand, days: int):
    """Servicegrad und Lagerbestand je Ziehung und Politikpunkt.

    C: (B x n) Gewichte, lost/held: (P x n) Summen je Artikel, demand: (n,).
    Rueckgabe: fill, inv jeweils (B x P).
    """
    fill = 1.0 - (C @ lost.T) / (C @ demand)[:, None]
    inv = (C @ held.T) / (C.sum(axis=1)[:, None] * days)
    return fill, inv


def _columns(policies):
    names = list(dict.fromkeys(policies))
    return {p: [i for i, x in enumerate(policies) if x == p] for p in names}


def inventory_at_levels(fill, inv, policies, level: float) -> dict:
    """Lagerbestand bei festem Servicegrad: je Politik ein Vektor ueber die Ziehungen."""
    out = {}
    for pol, cols in _columns(policies).items():
        out[pol] = np.array([inventory_at_service_level(fill[b, cols], inv[b, cols], level)
                             for b in range(fill.shape[0])])
    return out


def min_cost(C, lost, held, policies, ratio: float) -> dict:
    """Minimale Gesamtkosten je Politik (Parameter frei waehlbar) und Ziehung."""
    cost = C @ held.T + ratio * (C @ lost.T)
    return {pol: cost[:, cols].min(axis=1) for pol, cols in _columns(policies).items()}


def curve_bands(fill, inv, fill_point, policies, n_grid: int = 40, level: float = 0.95,
                min_valid: float = 0.9) -> dict:
    """Fehlerband der Kompromisskurve: Lagerbestand-Intervall bei gegebenem Servicegrad.

    Fuer jede Ziehung wird der Lagerbestand auf einem Raster von Servicegraden interpoliert;
    das Band sind die zentralen ``level`` der Ziehungen. Das Raster reicht von der kleinsten bis
    zur groessten Servicegrad-Punktschaetzung der Politik; Rasterpunkte, die in weniger als
    ``min_valid`` der Ziehungen erreichbar sind, werden weggelassen.

    Rueckgabe: {Politik: (Servicegrad-Raster, untere Grenze, obere Grenze)}.
    """
    a = (1 - level) / 2 * 100
    out = {}
    for pol, cols in _columns(policies).items():
        fp = np.asarray(fill_point)[cols]
        grid = np.linspace(fp.min(), fp.max(), n_grid)
        lo, hi, keep = [], [], []
        for g in grid:
            vals = np.array([inventory_at_service_level(fill[b, cols], inv[b, cols], g)
                             for b in range(fill.shape[0])])
            ok = ~np.isnan(vals)
            if ok.mean() >= min_valid:
                l, h_ = np.percentile(vals[ok], [a, 100 - a])
                lo.append(l)
                hi.append(h_)
                keep.append(g)
        out[pol] = (np.array(keep), np.array(lo), np.array(hi))
    return out
