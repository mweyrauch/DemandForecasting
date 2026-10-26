"""Rollierende Auswertung: Quantil-Modelle gegen Baselines, mit Bootstrap-Konfidenzintervallen.

Fuer jeden Cutoff (``evaluation.n_splits``, je ``horizon`` Tage Test) werden trainiert und
ausgewertet:
  * taegliche Quantil-Modelle (WAPE gegen gleitenden Durchschnitt),
  * Quantil-Modelle fuer die Bedarfssumme ueber Lieferzeit + 1 Tage (Pinball-Loss, Kalibrierung),
  * Bestandssimulation mit mehreren Politiken (siehe ``inventory.baselines`` in der Konfiguration):
      quantile_model_window[_floor]     Quantilmodell fuer die Bedarfssumme (optional mit Mindestbestand)
      baseline_mean_z[_floor]           Mittelwert der letzten 28 Tage + z * Sigma * sqrt(Fenster)
      baseline_empirical_q_<Tage>[_floor]  empirisches Quantil der Bedarfssummen im Rueckblick
Vergleich: Modell gegen die jeweils BESTE Baseline (Auswahl nach Punktschaetzung, damit konservativ
fuer das Modell). Unsicherheit: gepaarter Bootstrap ueber die Artikel.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dfi import plots
from dfi.baselines import moving_average
from dfi.bootstrap import (bootstrap_counts, curve_arrays, curve_bands, interval,
                           inventory_at_levels, min_cost)
from dfi.data import load_m5, resolve_data_dir, sales_matrix, to_long
from dfi.evaluate import rolling_origin_cutoffs
from dfi.features import add_window_target, build_features, required_history, window_train_mask
from dfi.inventory import (empirical_quantile_targets, inventory_at_service_level, policy_name,
                           simulate_order_up_to, window_sum)
from dfi.models import feature_importance_all, fit_quantile_models, predict_quantiles

MODEL = "quantile_model_window"
MODEL_FLOOR = "quantile_model_window_floor"


def pinball_elements(y, pred, q):
    d = y - pred
    return np.maximum(q * d, (q - 1) * d)


def fmt(point, lo, hi, pct=True):
    if pct:
        return f"{point:+.1%} [{lo:+.1%}, {hi:+.1%}]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stores", nargs="+", help="Filialen (ueberschreibt die Konfiguration)")
    ap.add_argument("--categories", nargs="+", help="Kategorien (ueberschreibt die Konfiguration)")
    ap.add_argument("--max-series", type=int, help="maximale Artikelzahl (ueberschreibt die Konfiguration)")
    ap.add_argument("--n-splits", type=int, help="Anzahl rollierender Cutoffs")
    ap.add_argument("--step-days", type=int, help="Abstand der Cutoffs in Tagen (Standard: horizon)")
    ap.add_argument("--out-dir", default="results", help="Ausgabeordner fuer Tabellen und Abbildungen")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.stores:
        cfg["data"]["stores"] = args.stores
    if args.categories:
        cfg["data"]["categories"] = args.categories
    if args.max_series:
        cfg["data"]["max_series"] = args.max_series
    if args.n_splits:
        cfg["evaluation"]["n_splits"] = args.n_splits
    if args.step_days:
        cfg["evaluation"]["step_days"] = args.step_days
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d, f, inv, ev = cfg["data"], cfg["forecast"], cfg["inventory"], cfg["evaluation"]
    h, qs = f["horizon"], f["quantiles"]
    L = inv["lead_time"]
    w = L + 1                     # Schutzzeitraum: Lieferzeit + 1 Tag
    qg = inv["quantile_grid"]
    n_splits = ev["n_splits"]
    step = int(ev.get("step_days") or h)
    min_stock = float(inv.get("min_stock", 1))
    specs = inv["baselines"]
    base_names = [policy_name(s) for s in specs]
    model_names = [MODEL] + ([MODEL_FLOOR] if inv.get("model_floor") else [])
    plots.set_policies(model_names + base_names)
    lookbacks = sorted({28} | {int(s["days"]) for s in specs if s["type"] == "empirical"})

    sales, calendar, prices = load_m5(resolve_data_dir(d), d["sales_file"], d["stores"],
                                      d["categories"], d["max_series"], d["seed"])
    ids, mat, day_cols = sales_matrix(sales)
    n, n_days = mat.shape

    # Historie: aeltester Cutoff - Trainingsfenster - Vorlauf fuer Lags und Rollfenster
    long_windows = tuple(f.get("long_windows", (91, 182, 364)))
    history = f["train_days"] + required_history(h, long_windows) + h + (n_splits - 1) * step
    long = to_long(sales, calendar, prices, last_days=history)
    print(f"Serien: {n}, Zeilen im Long-Format: {len(long):,}, Splits: {n_splits}")
    long, feats = build_features(long, horizon=h, long_windows=long_windows)
    long = add_window_target(long, L)
    # Speicher sparen: nur noch benoetigte Spalten behalten (Zeilen bleiben nach Artikel und Datum
    # sortiert) und alle Merkmale als float32 ablegen.
    long = long[["date", "sales", "target_window"] + feats]
    for col in feats:
        long[col] = long[col].astype("float32")
    gc.collect()
    dates = pd.to_datetime(calendar.set_index("d").loc[day_cols, "date"]).to_numpy()
    lag_col = f"lag_{h}"
    cutoffs = rolling_origin_cutoffs(n_days, h, n_splits, step)

    # ---- Akkumulatoren (je Artikel, ueber alle Splits)
    err_ma, err_m, act = np.zeros(n), np.zeros(n), np.zeros(n)
    pin = {}                                    # Politik -> {q: Array je Artikel}
    pin_cnt = np.zeros(n)
    calib_hits = {q: 0 for q in qg}
    calib_tot = 0
    daily_hits = {q: 0 for q in qs}
    daily_pin = {q: 0.0 for q in qs}
    daily_tot = 0
    points = None
    lost_s, held_s, dem_s = [], [], []
    imp_d_s, imp_w_s = [], []
    zero_series = {k: 0.0 for k in lookbacks}   # Anteil Artikel ohne Verkauf im Rueckblick
    zero_demand = {k: 0.0 for k in lookbacks}   # Testnachfrage dieser Artikel
    total_demand = 0.0

    for s, c in enumerate(cutoffs, 1):
        print(f"\n--- Split {s}/{len(cutoffs)}: Testfenster {str(dates[c])[:10]} bis "
              f"{str(dates[c + h - 1])[:10]} ---", flush=True)
        cutoff, end = dates[c], dates[c + h - 1]
        test = long[(long["date"] >= cutoff) & (long["date"] <= end)]
        in_train = (long["date"] >= cutoff - np.timedelta64(f["train_days"], "D")) & \
                   (long["date"] < cutoff)
        train_d = long[in_train].dropna(subset=[lag_col])
        train_w = long[in_train & window_train_mask(long["date"], cutoff, L)].dropna(
            subset=[lag_col, "target_window"])
        train_mat = mat[:, :c]
        y = mat[:, c:c + h].astype(float)

        # Diagnose: Artikel ohne Verkauf im Rueckblick (Zielniveau 0 bei den Baselines)
        for k in lookbacks:
            zero = train_mat[:, -k:].sum(axis=1) == 0
            zero_series[k] += zero.mean() / len(cutoffs)
            zero_demand[k] += y[zero].sum()
        total_demand += y.sum()

        # Modell 1: taegliche Quantile
        md = fit_quantile_models(train_d[feats], train_d["sales"], qs, f["backend"], f["model_params"])
        imp_d_s.append(feature_importance_all(md, feats, f["backend"], train_d, train_d["sales"]))
        pd_ = predict_quantiles(md, test[feats])
        P = {q: v.reshape(n, h) for q, v in pd_.items()}
        ma = moving_average(train_mat, h, 28)
        err_ma += np.abs(y - ma).sum(axis=1)
        err_m += np.abs(y - P[0.5]).sum(axis=1)
        act += np.abs(y).sum(axis=1)
        for q in qs:
            daily_hits[q] += int((y <= P[q]).sum())
            daily_pin[q] += float(pinball_elements(y, P[q], q).sum())
        daily_tot += y.size

        # Modell 2: Bedarfssumme ueber Lieferzeit + 1 Tage
        mw = fit_quantile_models(train_w[feats], train_w["target_window"], qg,
                                 f["backend"], f["model_params"])
        imp_w_s.append(feature_importance_all(mw, feats, f["backend"], train_w, train_w["target_window"]))
        PW = {q: v.reshape(n, h) for q, v in predict_quantiles(mw, test[feats]).items()}
        yw = test["target_window"].to_numpy().reshape(n, h)
        valid = ~np.isnan(yw)
        ywz = np.where(valid, yw, 0.0)
        pin_cnt += valid.sum(axis=1)
        calib_tot += int(valid.sum())
        for q in qg:
            calib_hits[q] += int(((yw <= PW[q]) & valid).sum())
        sigma = train_mat[:, -28:].std(axis=1)

        # Zielniveaus aller Politiken: (Name, Parameter, Ziel [n x h], ist_quantil)
        targets = []
        for q in qg:
            targets.append((MODEL, q, PW[q], True))
            if MODEL_FLOOR in model_names:
                targets.append((MODEL_FLOOR, q, np.maximum(PW[q], min_stock), True))
        for spec, name in zip(specs, base_names):
            floor = min_stock if spec.get("floor") else 0.0
            if spec["type"] == "mean_z":
                for z in inv["z_grid"]:
                    t_ = np.array([window_sum(ma[i], w) + z * sigma[i] * np.sqrt(w) for i in range(n)])
                    targets.append((name, z, np.maximum(t_, floor), False))
            else:
                for q in qg:
                    emp = empirical_quantile_targets(train_mat, w, q, h, int(spec["days"]))
                    targets.append((name, q, np.maximum(emp, floor), True))

        for name, q, tgt, is_q in targets:
            if is_q:
                pin.setdefault(name, {}).setdefault(q, np.zeros(n))
                pin[name][q] += (pinball_elements(ywz, tgt, q) * valid).sum(axis=1)

        if points is None:
            points = [(nm, a) for nm, a, _, _ in targets]
        lost = np.zeros((len(targets), n))
        held = np.zeros((len(targets), n))
        for k, (_, _, tgt, _) in enumerate(targets):
            for i in range(n):
                r = simulate_order_up_to(y[i], tgt[i], L)
                lost[k, i], held[k, i] = r["lost_units"], r["inventory_units_days"]
        lost_s.append(lost)
        held_s.append(held)
        dem_s.append(y.sum(axis=1))
        # Speicher freigeben, bevor der naechste Split neue Kopien anlegt
        del md, mw, train_d, train_w, test, targets, PW, P, pd_
        gc.collect()

    policies = np.array([p for p, _ in points])
    params = np.array([a for _, a in points])
    lost = np.sum(lost_s, axis=0)
    held = np.sum(held_s, axis=0)
    demand = np.sum(dem_s, axis=0)
    days = h * len(cutoffs)
    label = plots.policy_label

    # ------------------------------------------------------------------ Auswertung
    rows = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def add(metric, comparison, point, lo, hi):
        rows.append({"metric": metric, "comparison": comparison, "estimate": point,
                     "ci_low": lo, "ci_high": hi})

    C1 = np.ones((1, n))
    C = bootstrap_counts(n, ev["n_bootstrap"], ev["bootstrap_seed"])
    hist_panels = []

    fill1, inv1 = curve_arrays(C1, lost, held, demand, days)
    fillB, invB = curve_arrays(C, lost, held, demand, days)
    curve = pd.DataFrame({"policy": policies, "param": params,
                          "fill_rate": fill1[0], "avg_inventory": inv1[0]})
    curve.to_csv(out_dir / "inventory_tradeoff.csv", index=False)

    print("\n" + "=" * 70)
    print("Diagnose: Artikel ohne Verkauf im Rueckblick (Baselines haben dann Zielniveau 0,")
    print("ausser bei Mindestbestand). Gepoolt ueber die Splits.")
    for k in lookbacks:
        print(f"  Rueckblick {k:>3} Tage: {zero_series[k]:6.1%} der Artikel, "
              f"{zero_demand[k] / total_demand:6.2%} der Testnachfrage")
    print("Maximal erreichter Servicegrad je Politik (ueber das gesamte Raster):")
    diag = []
    for pol in plots.POLICY_ORDER:
        mx = curve.loc[curve["policy"] == pol, "fill_rate"].max()
        print(f"  {label(pol):<42} {mx:7.2%}")
        diag.append({"policy": pol, "max_fill_rate": mx})
    pd.DataFrame(diag).to_csv(out_dir / "max_fill_rate.csv", index=False)

    print("\n" + "=" * 70)
    print(f"Taegliche Prognose (gepoolt ueber {len(cutoffs)} Splits)")
    wape = lambda Cm, e: (Cm @ e) / (Cm @ act)          # noqa: E731
    w_ma, w_m = wape(C1, err_ma)[0], wape(C1, err_m)[0]
    wB_ma, wB_m = wape(C, err_ma), wape(C, err_m)
    imp = 1 - wB_m / wB_ma
    p0, (lo, hi) = 1 - w_m / w_ma, interval(imp)
    print(f"WAPE gleitender Durchschnitt: {w_ma:.4f} | WAPE Modell q50: {w_m:.4f}")
    print(f"Relative Verbesserung des WAPE: {fmt(p0, lo, hi)}")
    add("wape_improvement", "model_q50 vs moving_average_28", p0, lo, hi)
    plots.plot_wape({"gleitender Durchschnitt (28 Tage)": (w_ma, *interval(wB_ma)),
                     "Quantilmodell (q50)": (w_m, *interval(wB_m))}, fig_dir / "wape.png")
    hist_panels.append(("Tägliche Prognose: relative WAPE-Verbesserung", "Verbesserung des WAPE",
                        [("Quantilmodell vs. gleitender Durchschnitt", MODEL, imp, p0)]))
    for q in qs:
        print(f"q={q:<4} Kalibrierung={daily_hits[q] / daily_tot:.3f}  "
              f"Pinball={daily_pin[q] / daily_tot:.4f} (ohne Vergleichswert)")

    # ---- Pinball-Loss der Bedarfssumme: Modell vs. beste empirische Baseline je Quantil
    emp_names = [nm for nm in pin if nm.startswith("baseline_empirical_q")]
    print("\n" + "=" * 70)
    print(f"Bedarfssumme ueber {w} Tage: Pinball-Loss Modell vs. beste empirische Baseline")
    print("(positiv = Modell besser; Intervall: 95 %-Bootstrap ueber Artikel)")
    pin_series = {m: [] for m in model_names}
    pin_draws = {}
    for q in qg:
        best = min(emp_names, key=lambda nm: (C1 @ pin[nm][q])[0])
        pe, peB = (C1 @ pin[best][q])[0], C @ pin[best][q]
        text = f"q={q:<5} Kalibrierung={calib_hits[q] / calib_tot:.3f}  beste Baseline: {label(best)}"
        for m in model_names:
            pm, pmB = (C1 @ pin[m][q])[0], C @ pin[m][q]
            imp = 1 - pmB / peB
            p0, (lo, hi) = 1 - pm / pe, interval(imp)
            text += f"\n         {label(m):<32} {fmt(p0, lo, hi)}"
            add("pinball_improvement", f"{m} vs best empirical ({best}), q={q}", p0, lo, hi)
            pin_series[m].append((q, p0, lo, hi))
            pin_draws[(m, q)] = (imp, p0)
        print(text)
    plots.plot_pinball([(m, f"{label(m)} vs. beste empirische Baseline", pin_series[m])
                        for m in model_names], fig_dir / "pinball_improvement.png")
    q_hist = min(qg, key=lambda x: abs(x - 0.9))
    hist_panels.append((f"Bedarfssumme: Pinball-Verbesserung bei q={q_hist:g}", "Verbesserung des Pinball-Loss",
                        [(label(m), m, *pin_draws[(m, q_hist)]) for m in model_names]))

    # ---- Lagerbestand bei festem Servicegrad
    print("\n" + "=" * 70)
    print("Lagerbestand bei festem Servicegrad (Einheiten je Artikel; Vergleich mit der besten Baseline)")
    sl_plot = {}
    sl_hist_level = 0.95 if 0.95 in inv["service_levels"] else inv["service_levels"][len(inv["service_levels"]) // 2]
    best_at_hist = None
    for sl in inv["service_levels"]:
        p1 = inventory_at_levels(fill1, inv1, policies, sl)
        pB = inventory_at_levels(fillB, invB, policies, sl)
        print(f"\nServicegrad {sl:.0%}")
        # Nur auswerten, wenn der Zielservicegrad in der Punktschaetzung und in >= 90 % der
        # Ziehungen im simulierten Bereich liegt (sonst waere das Intervall irrefuehrend).
        ok = {pol: (not np.isnan(p1[pol][0])) and float(np.mean(~np.isnan(pB[pol]))) >= 0.9
              for pol in p1}
        sl_plot[sl] = {}
        for pol in plots.POLICY_ORDER:
            if ok[pol]:
                lo, hi = interval(pB[pol])
                print(f"  {label(pol):<42} {p1[pol][0]:7.2f}  [{lo:.2f}, {hi:.2f}]")
                add("inventory_at_service_level", f"{pol}, SL={sl}", p1[pol][0], lo, hi)
                sl_plot[sl][pol] = (p1[pol][0], lo, hi)
            else:
                share = float(np.mean(~np.isnan(pB[pol])))
                print(f"  {label(pol):<42} nicht erreicht (in {share:.0%} der Ziehungen erreichbar)")
        reach = [b for b in base_names if ok[b]]
        if not reach:
            print("  Vergleich: keine Baseline erreicht diesen Servicegrad")
            continue
        best = min(reach, key=lambda b: p1[b][0])
        print(f"  Beste Baseline: {label(best)}")
        series = []
        for m in model_names:
            if not ok[m]:
                continue
            rel = pB[m] / pB[best] - 1
            p0, (lo, hi) = p1[m][0] / p1[best][0] - 1, interval(rel)
            print(f"  {label(m)} vs. beste Baseline: {fmt(p0, lo, hi)}")
            add("inventory_rel_diff", f"{m} vs best baseline ({best}), SL={sl}", p0, lo, hi)
            series.append((f"{label(m)} vs. {label(best)}", m, rel, p0))
        if series and sl == sl_hist_level:
            hist_panels.append((f"Lagerbestand bei {sl:.0%} Servicegrad: relative Differenz",
                                "relative Differenz des Lagerbestands (negativ = Modell besser)", series))
            best_at_hist = best
    plots.plot_inventory_at_sl(sl_plot, fig_dir / "inventory_at_service_level.png")

    # ---- Kompromisskurve mit Fehlerband (Modelle und beste Baseline bei 95 %)
    if best_at_hist is None:
        reach_pts = {b: curve.loc[curve["policy"] == b, "fill_rate"].max() for b in base_names}
        best_at_hist = max(reach_pts, key=reach_pts.get)
    sel = set(model_names[:1] + [best_at_hist])
    cols = [i for i, p in enumerate(policies) if p in sel]
    bands = curve_bands(fillB[:, cols], invB[:, cols], fill1[0][cols], policies[cols])
    plots.plot_tradeoff(curve, fig_dir / "inventory_tradeoff.png", bands=bands)

    # ---- Kosten
    print("\n" + "=" * 70)
    print("Minimale Gesamtkosten je Kostenverhaeltnis (negativ = Modell guenstiger als beste Baseline)")
    cost_plot = {}
    r_hist = 20 if 20 in inv["cost_ratios"] else inv["cost_ratios"][len(inv["cost_ratios"]) // 2]
    for r in inv["cost_ratios"]:
        c1 = min_cost(C1, lost, held, policies, r)
        cB = min_cost(C, lost, held, policies, r)
        best = min(base_names, key=lambda b: c1[b][0])
        cost_plot[r] = {}
        series = []
        for m in model_names:
            rel = cB[m] / cB[best] - 1
            p0, (lo, hi) = c1[m][0] / c1[best][0] - 1, interval(rel)
            print(f"r={r:<4} {label(m):<32} vs. {label(best):<38} {fmt(p0, lo, hi)}")
            add("cost_rel_diff", f"{m} vs best baseline ({best}), r={r}", p0, lo, hi)
            cost_plot[r][m] = (p0, lo, hi)
            series.append((f"{label(m)} vs. {label(best)}", m, rel, p0))
        if r == r_hist:
            hist_panels.append((f"Gesamtkosten bei Kostenverhältnis r={r}: relative Differenz",
                                "relative Kostendifferenz (negativ = Modell günstiger)", series))
    plots.plot_costs(cost_plot, fig_dir / "cost_sensitivity.png")

    # ---- Stabilitaet
    print("\n" + "=" * 70)
    print("Stabilitaet ueber die Splits: Lagerbestand bei 95 % Servicegrad")
    stab = []
    for s_ in range(len(cutoffs)):
        f_s = 1 - lost_s[s_].sum(axis=1) / dem_s[s_].sum()
        i_s = held_s[s_].sum(axis=1) / (n * h)
        row = {"split": s_ + 1}
        for pol in dict.fromkeys(policies):
            cols_ = np.where(policies == pol)[0]
            row[pol] = inventory_at_service_level(f_s[cols_], i_s[cols_], 0.95)
        stab.append(row)
    stab_df = pd.DataFrame(stab)
    print(stab_df.rename(columns=label).round(2).to_string(index=False))
    plots.plot_stability(stab_df, fig_dir / "stability_splits.png")

    # ---- Merkmalswichtigkeit
    print("\n" + "=" * 70)
    print("Merkmalswichtigkeit (Anteil an der Gesamtwichtigkeit, Mittel ueber die Splits)")

    def aggregate(per_split):
        out = {}
        for q in per_split[0]:
            arr = np.vstack([s_[q] for s_ in per_split])
            out[q] = (np.array(feats), arr.mean(axis=0), arr.min(axis=0), arr.max(axis=0))
        return out

    imp_w, imp_d = aggregate(imp_w_s), aggregate(imp_d_s)
    imp_rows = []
    for model_name, imp_ in (("bedarfssumme", imp_w), ("taeglich", imp_d)):
        for q, (names, mean, lo_, hi_) in imp_.items():
            for nme, m_, l_, h_ in zip(names, mean, lo_, hi_):
                imp_rows.append({"model": model_name, "quantile": q, "feature": nme,
                                 "importance_mean": m_, "importance_min": l_, "importance_max": h_})
    pd.DataFrame(imp_rows).to_csv(out_dir / "feature_importance.csv", index=False)
    q_show = sorted({qg[0], min(qg, key=lambda x: abs(x - 0.9)), min(qg, key=lambda x: abs(x - 0.99))})
    ref = min(q_show, key=lambda x: abs(x - 0.9))
    names, mean, _, _ = imp_w[ref]
    for i in np.argsort(mean)[::-1][:8]:
        print(f"  {plots.feature_label(names[i]):<45} {mean[i]:6.1%}")
    plots.plot_feature_importance({q: imp_w[q] for q in q_show}, fig_dir / "feature_importance.png",
                                  title=f"Merkmalswichtigkeit: Bedarfssumme über {w} Tage")
    plots.plot_feature_importance({q: imp_d[q] for q in qs}, fig_dir / "feature_importance_daily.png",
                                  title="Merkmalswichtigkeit: tägliche Prognose")

    plots.plot_bootstrap_histograms(hist_panels, fig_dir / "bootstrap_histograms.png")
    pd.DataFrame(rows).to_csv(out_dir / "summary_ci.csv", index=False)
    info = {"stores": d["stores"], "categories": d["categories"], "n_series": int(n),
            "horizon": h, "step_days": step, "n_splits": len(cutoffs),
            "test_periods": [[str(dates[c])[:10], str(dates[c + h - 1])[:10]] for c in cutoffs]}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"\nErgebnisse gespeichert in {out_dir}/")


if __name__ == "__main__":
    main()
