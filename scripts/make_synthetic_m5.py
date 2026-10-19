"""Erzeugt kleine synthetische Daten im M5-Format (zum Testen und fuer die CI)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n_items", type=int, default=40)
    ap.add_argument("--n_days", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2013-01-01", periods=a.n_days)
    cal = pd.DataFrame({
        "date": dates,
        "wm_yr_wk": 11101 + (np.arange(a.n_days) // 7),
        "weekday": dates.day_name(),
        "wday": dates.weekday + 1,
        "month": dates.month,
        "year": dates.year,
        "d": [f"d_{i + 1}" for i in range(a.n_days)],
        "event_name_1": np.where(rng.random(a.n_days) < 0.03, "Event", None),
        "event_type_1": None,
        "event_name_2": None,
        "event_type_2": None,
        "snap_CA": (rng.random(a.n_days) < 0.33).astype(int),
        "snap_TX": 0,
        "snap_WI": 0,
    })
    cal.to_csv(out / "calendar.csv", index=False)

    rows, price_rows = [], []
    weekly = np.array([1.2, 1.3, 1.0, 0.9, 0.9, 1.0, 1.1])
    for i in range(a.n_items):
        item = f"FOODS_1_{i + 1:03d}"
        base_rate = rng.lognormal(mean=0.8, sigma=0.9)
        base_price = rng.uniform(1, 10)
        for store in ("CA_1", "CA_2"):
            season = weekly[dates.weekday % 7] * (1 + 0.2 * np.sin(np.arange(a.n_days) / 58))
            lam = base_rate * season * rng.uniform(0.7, 1.3)
            sales = rng.poisson(lam)
            rows.append({"id": f"{item}_{store}_evaluation", "item_id": item,
                         "dept_id": "FOODS_1", "cat_id": "FOODS", "store_id": store,
                         "state_id": "CA", **{f"d_{j + 1}": int(v) for j, v in enumerate(sales)}})
            for wk in cal["wm_yr_wk"].unique():
                price_rows.append((store, item, int(wk), round(base_price * rng.uniform(0.95, 1.05), 2)))

    pd.DataFrame(rows).to_csv(out / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(price_rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(
        out / "sell_prices.csv", index=False)
    print(f"Synthetische Daten geschrieben nach {out}")


if __name__ == "__main__":
    main()
