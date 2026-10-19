"""Baselines mit rollierender Zeitreihen-Validierung (Seasonal Naive, Moving Average)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from dfi.baselines import moving_average, seasonal_naive
from dfi.data import load_m5, resolve_data_dir, sales_matrix
from dfi.evaluate import mase, rolling_origin_cutoffs, wape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d, f = cfg["data"], cfg["forecast"]

    sales, _, _ = load_m5(resolve_data_dir(d), d["sales_file"], d["stores"], d["categories"],
                          d["max_series"], d["seed"])
    _, mat, _ = sales_matrix(sales)
    h = f["horizon"]
    cutoffs = rolling_origin_cutoffs(mat.shape[1], h, cfg["evaluation"]["n_splits"])

    rows = []
    for c in cutoffs:
        train, test = mat[:, :c], mat[:, c:c + h]
        for name, fc in (("seasonal_naive", seasonal_naive(train, h)),
                         ("moving_average_28", moving_average(train, h, 28))):
            rows.append({"cutoff": c, "model": name, "wape": wape(test, fc),
                         "mase": mase(test, fc, train, season=7)})
    res = pd.DataFrame(rows)
    print(res.groupby("model")[["wape", "mase"]].agg(["mean", "std"]).round(4))
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/baselines.csv", index=False)


if __name__ == "__main__":
    main()
