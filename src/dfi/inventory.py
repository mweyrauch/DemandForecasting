"""Einfache Bestandssimulation (Order-up-to-Politik, taegliche Disposition).

ANNAHMEN (bewusst vereinfacht, in der README offen benennen):
- feste Lieferzeit, keine Mindestbestellmengen, keine Kapazitaetsgrenzen
- nicht gedeckte Nachfrage geht verloren (Lost Sales)
- Kostensaetze sind Platzhalter und keine realen Werte
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def window_sum(fc: np.ndarray, window: int) -> np.ndarray:
    """Summe der Prognose ueber die naechsten ``window`` Tage (am Ende abgeschnitten)."""
    fc = np.asarray(fc, float)
    csum = np.concatenate([[0.0], np.cumsum(fc)])
    idx = np.arange(len(fc))
    end = np.minimum(idx + window, len(fc))
    return csum[end] - csum[idx]


def simulate_order_up_to(demand, target_level, lead_time: int = 3,
                         holding_cost: float = 0.01, stockout_cost: float = 0.10) -> dict:
    """Simuliert eine Serie. ``target_level[t]`` ist der Order-up-to-Bestand an Tag t."""
    demand = np.asarray(demand, float)
    target = np.asarray(target_level, float)
    on_hand = float(target[0])
    pipeline = [0.0] * lead_time
    lost = held = 0.0
    for t, dem in enumerate(demand):
        if lead_time > 0:
            on_hand += pipeline.pop(0)
        sold = min(on_hand, dem)
        lost += dem - sold
        on_hand -= sold
        position = on_hand + sum(pipeline)
        order = max(0.0, target[t] - position)
        if lead_time > 0:
            pipeline.append(order)
        else:
            on_hand += order
        held += on_hand
    total = demand.sum()
    return {
        "fill_rate": 1.0 - lost / total if total > 0 else 1.0,
        "avg_inventory": held / len(demand),
        "inventory_units_days": held,
        "lost_units": lost,
        "total_cost": holding_cost * held + stockout_cost * lost,
    }


def empirical_quantile_targets(train: np.ndarray, window: int, q: float, n_days: int,
                               emp_days: int = 84) -> np.ndarray:
    """Starke Baseline ohne Modell: konstantes Zielniveau je Serie.

    Nimmt die rollierenden Bedarfssummen (Laenge ``window``) der letzten ``emp_days``
    Trainingstage und davon das empirische q-Quantil. Beruecksichtigt so auch sporadische
    Nachfrage, anders als "Mittelwert + z * Standardabweichung".
    """
    recent = train[:, -emp_days:]
    csum = np.cumsum(np.pad(recent, ((0, 0), (1, 0))), axis=1)
    sums = csum[:, window:] - csum[:, :-window]
    level = np.quantile(sums, q, axis=1)
    return np.repeat(level[:, None], n_days, axis=1)


def inventory_at_service_level(fill_rate, avg_inventory, target: float) -> float:
    """Lagerbestand bei vorgegebenem Servicegrad (lineare Interpolation der Kompromisskurve).

    Gibt NaN zurueck, wenn der Zielservicegrad ausserhalb der simulierten Punkte liegt
    (keine Extrapolation).
    """
    f = np.asarray(fill_rate, float)
    i = np.asarray(avg_inventory, float)
    order = np.argsort(f)
    f, i = f[order], i[order]
    if target < f[0] or target > f[-1]:
        return float("nan")
    return float(np.interp(target, f, i))


def best_cost_by_ratio(results: pd.DataFrame, ratios) -> pd.DataFrame:
    """Minimale Gesamtkosten je Politik und Kostenverhaeltnis.

    Kosten = Lagerbestand-Einheitentage + Verhaeltnis * entgangene Einheiten
    (Lagerkosten pro Einheit und Tag auf 1 normiert). Jede Politik darf ihren Parameter
    (z bzw. Quantil) fuer jedes Kostenverhaeltnis neu waehlen.
    """
    rows = []
    for ratio in ratios:
        cost = results["inventory_units_days"] + ratio * results["lost_units"]
        for policy, g in results.assign(cost=cost).groupby("policy"):
            best = g.loc[g["cost"].idxmin()]
            rows.append({"cost_ratio": ratio, "policy": policy, "best_param": best["param"],
                         "cost": best["cost"], "fill_rate": best["fill_rate"],
                         "avg_inventory": best["avg_inventory"]})
    return pd.DataFrame(rows)
