"""Quantil-Modelle trainieren, evaluieren und mit Baselines in der Bestandssimulation vergleichen.

Verglichen werden drei Politiken, jeweils ueber einen Parameterbereich (Kompromisskurve):
  1. baseline_mean_z:        Mittelwert der letzten 28 Tage + z * Sigma * sqrt(Fenster)
  2. baseline_empirical_q:   empirisches Quantil der Bedarfssummen der letzten Wochen
  3. quantile_model_window:  Quantilmodell, das die Bedarfssumme ueber Lieferzeit + 1 Tage
                             direkt vorhersagt

Vereinfachung im Skelett: ein einzelner Cutoff (letzte ``horizon`` Tage als Test).
TODO: auf rollierende Cutoffs erweitern (siehe evaluate.rolling_origin_cutoffs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from dfi.baselines import moving_average
from dfi.data import load_m5, resolve_data_dir, sales_matrix, to_long
from dfi.evaluate import calibration, pinball_loss, wape
from dfi.features import add_window_target, build_features, window_train_mask
from dfi.inventory import (best_cost_by_ratio, empirical_quantile_targets,
                           inventory_at_service_level, simulate_order_up_to, window_sum)
from dfi.models import fit_quantile_models, predict_quantiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    d, f, inv = cfg["data"], cfg["forecast"], cfg["inventory"]
    h, qs = f["horizon"], f["quantiles"]
    L = inv["lead_time"]
    w = L + 1                     # Schutzzeitraum: Lieferzeit + 1 Tag
    qg = inv["quantile_grid"]

    sales, calendar, prices = load_m5(resolve_data_dir(d), d["sales_file"], d["stores"],
                                      d["categories"], d["max_series"], d["seed"])
    ids, mat, day_cols = sales_matrix(sales)
    n = len(ids)

    # Benoetigt werden: Testfenster + Trainingsfenster + Vorlauf fuer Lags/Rolling (max. 56 Tage)
    history = f["train_days"] + h + 56
    long = to_long(sales, calendar, prices, last_days=history)
    print(f"Serien: {n}, Zeilen im Long-Format: {len(long):,}")
    long, feats = build_features(long, horizon=h)
    long = add_window_target(long, L)

    dates = pd.to_datetime(calendar.set_index("d").loc[day_cols, "date"]).to_numpy()
    cutoff = dates[-h]                                   # erster Testtag
    first_train = cutoff - np.timedelta64(f["train_days"], "D")
    lag_col = f"lag_{h}"
    test = long[long["date"] >= cutoff]
    in_train = (long["date"] >= first_train) & (long["date"] < cutoff)
    train_d = long[in_train].dropna(subset=[lag_col])
    in_train_w = in_train & window_train_mask(long["date"], cutoff, L)
    train_w = long[in_train_w].dropna(subset=[lag_col, "target_window"])

    # --- Modell 1: taegliche Quantile (nur fuer Genauigkeitsmetriken)
    models_d = fit_quantile_models(train_d[feats], train_d["sales"], qs, f["backend"], f["model_params"])
    pred_d = predict_quantiles(models_d, test[feats])
    y = test["sales"].to_numpy().reshape(n, h)
    P = {q: v.reshape(n, h) for q, v in pred_d.items()}
    train_mat = mat[:, :-h]
    ma = moving_average(train_mat, h, 28)

    print("\n== Taegliche Prognose (Testfenster) ==")
    print(f"WAPE moving_average_28: {wape(y, ma):.4f}")
    print(f"WAPE quantile q50:      {wape(y, P[0.5]):.4f}")
    for q in qs:
        print(f"q={q:<4} pinball={pinball_loss(y, P[q], q):.4f}  Kalibrierung={calibration(y, P[q]):.3f}")

    # --- Modell 2: Bedarfssumme ueber Lieferzeit + 1 Tage (direkt gelernt)
    models_w = fit_quantile_models(train_w[feats], train_w["target_window"], qg,
                                   f["backend"], f["model_params"])
    pred_w = predict_quantiles(models_w, test[feats])
    PW = {q: v.reshape(n, h) for q, v in pred_w.items()}
    yw = test["target_window"].to_numpy().reshape(n, h)
    valid = ~np.isnan(yw)
    print(f"\n== Bedarfssumme ueber {w} Tage (Kalibrierung, nur vollstaendige Fenster) ==")
    for q in qg:
        print(f"q={q:<5} Kalibrierung={calibration(yw[valid], PW[q][valid]):.3f}")

    # --- Bestandssimulation
    sigma = train_mat[:, -28:].std(axis=1)
    rows = []

    def run(policy, param, targets):
        res = [simulate_order_up_to(y[i], targets[i], L) for i in range(n)]
        lost = sum(r["lost_units"] for r in res)
        rows.append({
            "policy": policy, "param": param,
            "fill_rate": 1.0 - lost / y.sum(),                      # mengengewichtet
            "avg_inventory": float(np.mean([r["avg_inventory"] for r in res])),
            "inventory_units_days": sum(r["inventory_units_days"] for r in res),
            "lost_units": lost,
        })

    for z in inv["z_grid"]:
        run("baseline_mean_z", z,
            [window_sum(ma[i], w) + z * sigma[i] * np.sqrt(w) for i in range(n)])
    for q in qg:
        run("baseline_empirical_q", q,
            empirical_quantile_targets(train_mat, w, q, h, inv["empirical_days"]))
        run("quantile_model_window", q, PW[q])

    res = pd.DataFrame(rows)
    print("\n== Kompromisskurven (Servicegrad vs. Lagerbestand) ==")
    print(res.drop(columns=["inventory_units_days", "lost_units"]).round(3).to_string(index=False))

    # --- Vergleich bei festem Servicegrad
    sl_rows = []
    for sl in inv["service_levels"]:
        for pol, g in res.groupby("policy"):
            sl_rows.append({"service_level": sl, "policy": pol,
                            "avg_inventory": inventory_at_service_level(
                                g["fill_rate"], g["avg_inventory"], sl)})
    sl_df = pd.DataFrame(sl_rows)
    sl_wide = sl_df.pivot(index="service_level", columns="policy", values="avg_inventory")
    print("\n== Durchschnittlicher Lagerbestand bei festem Servicegrad (NaN = nicht erreicht) ==")
    print(sl_wide.round(2).to_string())

    # --- Kostensensitivitaet
    cost_df = best_cost_by_ratio(res, inv["cost_ratios"])
    cost_wide = cost_df.pivot(index="cost_ratio", columns="policy", values="cost")
    print("\n== Minimale Gesamtkosten je Kostenverhaeltnis (Lager-Einheitentage + r * Fehlmenge) ==")
    print(cost_wide.round(0).to_string())

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    res.to_csv("results/inventory_tradeoff.csv", index=False)
    sl_df.to_csv("results/inventory_at_service_level.csv", index=False)
    cost_df.to_csv("results/cost_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for pol, g in res.groupby("policy"):
        g = g.sort_values("fill_rate")
        ax.plot(g["avg_inventory"], g["fill_rate"], marker="o", label=pol)
    ax.set_xlabel("durchschnittlicher Lagerbestand (Einheiten/Serie)")
    ax.set_ylabel("Servicegrad (Fill Rate)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/inventory_tradeoff.png", dpi=150)


if __name__ == "__main__":
    main()
