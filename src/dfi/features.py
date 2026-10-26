"""Feature Engineering ohne Zukunftsinformation.

Kernidee (direkte Mehrschritt-Strategie): Alle Lag- und Rolling-Features sind um
mindestens ``horizon`` Tage verschoben. Damit stehen sie fuer jeden Tag im
Prognosefenster zum Zeitpunkt der Prognose tatsaechlich zur Verfuegung -
kein rekursives Vorhersagen und kein Data Leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAT_FEATURES = ["item_code", "store_code", "dept_code"]


def required_history(horizon: int, long_windows=()) -> int:
    """Tage Vorlauf, die vor dem ersten Trainingstag geladen werden muessen."""
    return max(56, horizon + max(long_windows, default=0))


def build_features(df: pd.DataFrame, horizon: int = 28,
                   long_windows=(91, 182, 364)) -> tuple[pd.DataFrame, list[str]]:
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    g = df.groupby("id")["sales"]

    feats: list[str] = []
    for lag in (horizon, horizon + 7, horizon + 14, horizon + 28):
        name = f"lag_{lag}"
        df[name] = g.shift(lag)
        feats.append(name)

    shifted = g.shift(horizon)
    for w in (7, 28):
        name = f"rmean_{horizon}_{w}"
        roll = shifted.groupby(df["id"]).rolling(w, min_periods=1).mean()
        df[name] = roll.reset_index(level=0, drop=True).astype("float32")
        feats.append(name)

    # Langfristmerkmale: gleiche Verschiebung um ``horizon`` Tage wie oben (kein Leakage)
    long_windows = tuple(sorted(long_windows or ()))
    if long_windows:
        by_id = df["id"]
        for wdw in long_windows:
            name = f"rmean_{horizon}_{wdw}"
            roll = shifted.groupby(by_id).rolling(wdw, min_periods=max(1, wdw // 2)).mean()
            df[name] = roll.reset_index(level=0, drop=True).astype("float32")
            feats.append(name)
        zero = (df["sales"] == 0).astype("float32").where(df["sales"].notna())
        zero_shifted = zero.groupby(by_id).shift(horizon)
        for wdw in sorted({long_windows[0], long_windows[-1]}):
            name = f"rzero_{horizon}_{wdw}"
            roll = zero_shifted.groupby(by_id).rolling(wdw, min_periods=max(1, wdw // 2)).mean()
            df[name] = roll.reset_index(level=0, drop=True).astype("float32")
            feats.append(name)
        name = f"rstd_{horizon}_{long_windows[0]}"
        roll = shifted.groupby(by_id).rolling(long_windows[0], min_periods=max(2, long_windows[0] // 2)).std()
        df[name] = roll.reset_index(level=0, drop=True).astype("float32")
        feats.append(name)

    # Preis ist zum Planungszeitpunkt bekannt -> kein Leakage
    df["price_rel"] = df["sell_price"] / df.groupby("id")["sell_price"].transform("mean")
    df["has_event"] = df["event_name_1"].notna().astype("int8")
    df["dayofmonth"] = df["date"].dt.day
    feats += ["sell_price", "price_rel", "has_event", "snap", "wday", "month", "dayofmonth"]

    for src, dst in (("item_id", "item_code"), ("store_id", "store_code"), ("dept_id", "dept_code")):
        df[dst] = df[src].astype("category").cat.codes.astype("int32")
    feats += CAT_FEATURES
    return df, feats


def add_window_target(df: pd.DataFrame, lead_time: int) -> pd.DataFrame:
    """Bedarfssumme ueber ``lead_time + 1`` Tage ab dem jeweiligen Tag (inkl. heute).

    Das ist die Groesse, die fuer das Order-up-to-Niveau relevant ist: Bestellt wird heute,
    die Ware kommt nach der Lieferzeit, und bis zur naechsten Lieferung muss der Bestand
    reichen. Am Datenende ist die Summe unvollstaendig (NaN).
    """
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    g = df.groupby("id")["sales"]
    total = df["sales"].copy()
    for k in range(1, lead_time + 1):
        total = total + g.shift(-k)
    df["target_window"] = total
    return df


def window_train_mask(dates: pd.Series, cutoff, lead_time: int) -> pd.Series:
    """Trainingszeilen fuer das Fenstersummen-Modell.

    Das Zielfenster ``[t, t + lead_time]`` muss komplett VOR dem Cutoff enden, sonst fliesst
    Information aus dem Testzeitraum ins Training (Leakage).
    """
    last_start = np.datetime64(cutoff) - np.timedelta64(lead_time + 1, "D")
    return dates <= last_start
