"""Laden und Aufbereiten der M5-Daten."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAL_COLS = ["d", "date", "wm_yr_wk", "wday", "month", "year", "event_name_1",
            "snap_CA", "snap_TX", "snap_WI"]


def resolve_data_dir(data_cfg: dict) -> Path:
    """Gibt den Ordner mit den M5-Dateien zurueck.

    ``source: local``     -> ``data_cfg["dir"]``
    ``source: kagglehub`` -> Download (bzw. Cache) ueber das Paket ``kagglehub``.
    Die Dateien landen dabei im kagglehub-Cache ausserhalb des Repositories.
    """
    source = data_cfg.get("source", "local")
    if source == "local":
        return Path(data_cfg["dir"])
    if source != "kagglehub":
        raise ValueError(f"Unbekannte Datenquelle: {source}")

    try:
        import kagglehub
    except ImportError as e:  # pragma: no cover
        raise ImportError('kagglehub fehlt: pip install -e ".[kaggle]"') from e

    kh = data_cfg.get("kagglehub", {})
    handle = kh.get("handle", "m5-forecasting-accuracy")
    if kh.get("kind", "competition") == "competition":
        # Voraussetzung: Kaggle-Zugangsdaten und akzeptierte Wettbewerbsregeln
        root = Path(kagglehub.competition_download(handle))
    else:
        root = Path(kagglehub.dataset_download(handle))

    hits = list(root.rglob("calendar.csv"))
    if not hits:
        raise FileNotFoundError(f"calendar.csv nicht gefunden unter {root}")
    return hits[0].parent


def day_columns(sales: pd.DataFrame) -> list[str]:
    cols = [c for c in sales.columns if c.startswith("d_")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def load_m5(data_dir, sales_file="sales_train_evaluation.csv", stores=None, categories=None,
            max_series=None, seed=0):
    """Liest die drei M5-Dateien und waehlt optional eine Teilmenge der Zeitreihen.

    Ist ``max_series`` kleiner als die Zahl der gefilterten Serien, wird nach Abteilung
    (``dept_id``) geschichtet gezogen, damit alle Abteilungen anteilig vertreten bleiben.
    ``max_series=None`` nutzt alle Serien des Filters.
    """
    data_dir = Path(data_dir)
    path = data_dir / sales_file
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: "int16" for c in header if c.startswith("d_")}  # spart ~75 % Speicher
    sales = pd.read_csv(path, dtype=dtypes)
    if stores:
        sales = sales[sales["store_id"].isin(stores)]
    if categories:
        sales = sales[sales["cat_id"].isin(categories)]
    if max_series and len(sales) > max_series:
        frac = max_series / len(sales)
        sales = sales.groupby("dept_id", group_keys=False).sample(frac=frac, random_state=seed)
    sales = sales.sort_values("id").reset_index(drop=True)
    calendar = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    return sales, calendar, prices


def sales_matrix(sales: pd.DataFrame):
    """Gibt (ids, Matrix [Serien x Tage], Tagesspalten) zurueck."""
    cols = day_columns(sales)
    return sales["id"].to_numpy(), sales[cols].to_numpy(dtype=np.float32), cols


def to_long(sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame,
            last_days: int | None = None) -> pd.DataFrame:
    """Wide -> long und Zusammenfuehren mit Kalender und Preisen.

    ``last_days`` behaelt nur die letzten N Tage (spart Speicher; fuer das Training werden
    nur Trainingsfenster + Testfenster + Lag-Vorlauf benoetigt).

    Hinweis: ``sell_price`` ist NaN, solange ein Artikel noch nicht im Sortiment war.
    """
    cols = day_columns(sales)
    if last_days:
        cols = cols[-last_days:]
    long = sales.melt(id_vars=ID_COLS, value_vars=cols, var_name="d", value_name="sales")
    cal = calendar[[c for c in CAL_COLS if c in calendar.columns]]
    long = long.merge(cal, on="d", how="left")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long["snap"] = np.select(
        [long["state_id"] == "CA", long["state_id"] == "TX", long["state_id"] == "WI"],
        [long["snap_CA"], long["snap_TX"], long["snap_WI"]], default=0)
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    long["sales"] = long["sales"].astype("float32")
    long["sell_price"] = long["sell_price"].astype("float32")
    return long.sort_values(["id", "date"]).reset_index(drop=True)
