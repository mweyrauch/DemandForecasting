"""Auswertung eines Laufs: Tabellen, Konsolenausgabe und Abbildungen.

Das Modell wird mit der jeweils BESTEN Baseline verglichen (Auswahl nach Punktschaetzung, damit
konservativ fuer das Modell). Unsicherheit: gepaarter Bootstrap ueber die Artikel.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .bootstrap import (
    bootstrap_counts,
    curve_arrays,
    curve_bands,
    interval,
    inventory_at_levels,
    min_cost,
)
from .inventory import inventory_at_service_level
from .pipeline import MODEL, Pooled, Prepared, Settings

RULE = "\n" + "=" * 70


def fmt(point, lo, hi) -> str:
    return f"{point:+.1%} [{lo:+.1%}, {hi:+.1%}]"


class Report:
    """Wertet die zusammengefassten Splits aus und schreibt alles in ``out_dir``."""

    def __init__(self, st: Settings, prep: Prepared, res: Pooled, out_dir):
        self.st, self.prep, self.res = st, prep, res
        self.out_dir = Path(out_dir)
        self.fig_dir = self.out_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []              # Zeilen fuer summary_ci.csv
        self.hist_panels = []       # Panels fuer die Bootstrap-Histogramme
        self.best_at_hist = None    # beste Baseline beim Servicegrad des Histogramms

        ev = st.cfg["evaluation"]
        n = prep.n_series
        days = st.horizon * res.n_splits
        self.C1 = np.ones((1, n))                                   # Punktschaetzung
        self.C = bootstrap_counts(n, ev["n_bootstrap"], ev["bootstrap_seed"])
        self.fill1, self.inv1 = curve_arrays(self.C1, res.lost, res.held, res.demand, days)
        self.fillB, self.invB = curve_arrays(self.C, res.lost, res.held, res.demand, days)
        self.curve = pd.DataFrame({"policy": res.policies, "param": res.params,
                                   "fill_rate": self.fill1[0], "avg_inventory": self.inv1[0]})

    def add(self, metric, comparison, point, lo, hi):
        self.rows.append({"metric": metric, "comparison": comparison, "estimate": point,
                          "ci_low": lo, "ci_high": hi})

    def run(self):
        plots.set_policies(self.st.policy_names)
        self.curve.to_csv(self.out_dir / "inventory_tradeoff.csv", index=False)
        self.diagnosis()
        self.daily_forecast()
        self.pinball()
        self.service_levels()
        self.tradeoff()
        self.costs()
        self.stability()
        self.feature_importance()
        plots.plot_bootstrap_histograms(self.hist_panels, self.fig_dir / "bootstrap_histograms.png")
        pd.DataFrame(self.rows).to_csv(self.out_dir / "summary_ci.csv", index=False)
        self.run_info()
        print(f"\nErgebnisse gespeichert in {self.out_dir}/")

    # ------------------------------------------------------------------ Abschnitte

    def diagnosis(self):
        res, label = self.res, plots.policy_label
        print(RULE)
        print("Diagnose: Artikel ohne Verkauf im Rueckblick (Baselines haben dann Zielniveau 0,")
        print("ausser bei Mindestbestand). Gepoolt ueber die Splits.")
        for k in self.st.lookbacks:
            print(f"  Rueckblick {k:>3} Tage: {res.zero_share[k]:6.1%} der Artikel, "
                  f"{res.zero_demand[k] / res.total_demand:6.2%} der Testnachfrage")
        print("Maximal erreichter Servicegrad je Politik (ueber das gesamte Raster):")
        diag = []
        for pol in plots.POLICY_ORDER:
            mx = self.curve.loc[self.curve["policy"] == pol, "fill_rate"].max()
            print(f"  {label(pol):<42} {mx:7.2%}")
            diag.append({"policy": pol, "max_fill_rate": mx})
        pd.DataFrame(diag).to_csv(self.out_dir / "max_fill_rate.csv", index=False)

    def daily_forecast(self):
        res, C1, C = self.res, self.C1, self.C
        print(RULE)
        print(f"Taegliche Prognose (gepoolt ueber {res.n_splits} Splits)")
        wape = lambda Cm, e: (Cm @ e) / (Cm @ res.actual)          # noqa: E731
        w_ma, w_m = wape(C1, res.err_ma)[0], wape(C1, res.err_model)[0]
        wB_ma, wB_m = wape(C, res.err_ma), wape(C, res.err_model)
        imp = 1 - wB_m / wB_ma
        p0, (lo, hi) = 1 - w_m / w_ma, interval(imp)
        print(f"WAPE gleitender Durchschnitt: {w_ma:.4f} | WAPE Modell q50: {w_m:.4f}")
        print(f"Relative Verbesserung des WAPE: {fmt(p0, lo, hi)}")
        self.add("wape_improvement", "model_q50 vs moving_average_28", p0, lo, hi)
        plots.plot_wape({"gleitender Durchschnitt (28 Tage)": (w_ma, *interval(wB_ma)),
                         "Quantilmodell (q50)": (w_m, *interval(wB_m))}, self.fig_dir / "wape.png")
        self.hist_panels.append((
            "Tägliche Prognose: relative WAPE-Verbesserung", "Verbesserung des WAPE",
            [("Quantilmodell vs. gleitender Durchschnitt", MODEL, imp, p0)]))
        for q in self.st.daily_quantiles:
            print(f"q={q:<4} Kalibrierung={res.daily_hits[q] / res.daily_total:.3f}  "
                  f"Pinball={res.daily_pinball[q] / res.daily_total:.4f} (ohne Vergleichswert)")

    def pinball(self):
        """Pinball-Loss der Bedarfssumme: Modell gegen die beste empirische Baseline je Quantil."""
        st, res, C1, C, label = self.st, self.res, self.C1, self.C, plots.policy_label
        pin, qg = res.pinball, st.quantile_grid
        emp_names = [nm for nm in pin if nm.startswith("baseline_empirical_q")]
        print(RULE)
        print(f"Bedarfssumme ueber {st.window} Tage: Pinball-Loss Modell vs. beste empirische Baseline")
        print("(positiv = Modell besser; Intervall: 95 %-Bootstrap ueber Artikel)")
        series = {m: [] for m in st.model_names}
        draws = {}
        for q in qg:
            best = min(emp_names, key=lambda nm: (C1 @ pin[nm][q])[0])
            pe, peB = (C1 @ pin[best][q])[0], C @ pin[best][q]
            text = (f"q={q:<5} Kalibrierung={res.calib_hits[q] / res.calib_total:.3f}  "
                    f"beste Baseline: {label(best)}")
            for m in st.model_names:
                pm, pmB = (C1 @ pin[m][q])[0], C @ pin[m][q]
                imp = 1 - pmB / peB
                p0, (lo, hi) = 1 - pm / pe, interval(imp)
                text += f"\n         {label(m):<32} {fmt(p0, lo, hi)}"
                self.add("pinball_improvement", f"{m} vs best empirical ({best}), q={q}", p0, lo, hi)
                series[m].append((q, p0, lo, hi))
                draws[(m, q)] = (imp, p0)
            print(text)
        plots.plot_pinball([(m, f"{label(m)} vs. beste empirische Baseline", series[m])
                            for m in st.model_names], self.fig_dir / "pinball_improvement.png")
        q_hist = min(qg, key=lambda x: abs(x - 0.9))
        self.hist_panels.append((
            f"Bedarfssumme: Pinball-Verbesserung bei q={q_hist:g}", "Verbesserung des Pinball-Loss",
            [(label(m), m, *draws[(m, q_hist)]) for m in st.model_names]))

    def service_levels(self):
        """Lagerbestand bei festem Servicegrad, Modell gegen die beste Baseline."""
        st, label = self.st, plots.policy_label
        levels = st.cfg["inventory"]["service_levels"]
        policies = self.res.policies
        print(RULE)
        print("Lagerbestand bei festem Servicegrad (Einheiten je Artikel; Vergleich mit der besten Baseline)")
        sl_plot = {}
        hist_level = 0.95 if 0.95 in levels else levels[len(levels) // 2]
        for sl in levels:
            p1 = inventory_at_levels(self.fill1, self.inv1, policies, sl)
            pB = inventory_at_levels(self.fillB, self.invB, policies, sl)
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
                    self.add("inventory_at_service_level", f"{pol}, SL={sl}", p1[pol][0], lo, hi)
                    sl_plot[sl][pol] = (p1[pol][0], lo, hi)
                else:
                    share = float(np.mean(~np.isnan(pB[pol])))
                    print(f"  {label(pol):<42} nicht erreicht (in {share:.0%} der Ziehungen erreichbar)")
            reach = [b for b in st.baseline_names if ok[b]]
            if not reach:
                print("  Vergleich: keine Baseline erreicht diesen Servicegrad")
                continue
            best = min(reach, key=lambda b: p1[b][0])
            print(f"  Beste Baseline: {label(best)}")
            series = []
            for m in st.model_names:
                if not ok[m]:
                    continue
                rel = pB[m] / pB[best] - 1
                p0, (lo, hi) = p1[m][0] / p1[best][0] - 1, interval(rel)
                print(f"  {label(m)} vs. beste Baseline: {fmt(p0, lo, hi)}")
                self.add("inventory_rel_diff", f"{m} vs best baseline ({best}), SL={sl}", p0, lo, hi)
                series.append((f"{label(m)} vs. {label(best)}", m, rel, p0))
            if series and sl == hist_level:
                self.hist_panels.append((
                    f"Lagerbestand bei {sl:.0%} Servicegrad: relative Differenz",
                    "relative Differenz des Lagerbestands (negativ = Modell besser)", series))
                self.best_at_hist = best
        plots.plot_inventory_at_sl(sl_plot, self.fig_dir / "inventory_at_service_level.png")

    def tradeoff(self):
        """Kompromisskurve mit Fehlerband fuer das Modell und die beste Baseline bei 95 %."""
        st, ev, label = self.st, self.st.cfg["evaluation"], plots.policy_label
        curve, policies = self.curve, self.res.policies
        best = self.best_at_hist
        if best is None:
            reach = {b: curve.loc[curve["policy"] == b, "fill_rate"].max() for b in st.baseline_names}
            best = max(reach, key=reach.get)
        sel = set(st.model_names[:1] + [best])
        cols = [i for i, p in enumerate(policies) if p in sel]
        bands = curve_bands(self.fillB[:, cols], self.invB[:, cols], self.fill1[0][cols],
                            policies[cols], min_valid=float(ev.get("band_min_valid", 0.3)))
        for pol in plots.POLICY_ORDER:
            if pol in bands:
                g_ = bands[pol][0]
                print(f"Fehlerband {label(pol)}: " + (
                    f"Servicegrad {g_.min():.1%} bis {g_.max():.1%}"
                    if len(g_) > 1 else "keine Rasterpunkte mit genug gueltigen Ziehungen"))
        plots.plot_tradeoff(curve, self.fig_dir / "inventory_tradeoff.png", bands=bands,
                            fill_max=ev.get("tradeoff_fill_max", 0.995))

    def costs(self):
        """Minimale Gesamtkosten je Kostenverhaeltnis, Modell gegen die beste Baseline."""
        st, res, label = self.st, self.res, plots.policy_label
        ratios = st.cfg["inventory"]["cost_ratios"]
        print(RULE)
        print("Minimale Gesamtkosten je Kostenverhaeltnis (negativ = Modell guenstiger als beste Baseline)")
        cost_plot = {}
        r_hist = 20 if 20 in ratios else ratios[len(ratios) // 2]
        for r in ratios:
            c1 = min_cost(self.C1, res.lost, res.held, res.policies, r)
            cB = min_cost(self.C, res.lost, res.held, res.policies, r)
            best = min(st.baseline_names, key=lambda b: c1[b][0])
            cost_plot[r] = {}
            series = []
            for m in st.model_names:
                rel = cB[m] / cB[best] - 1
                p0, (lo, hi) = c1[m][0] / c1[best][0] - 1, interval(rel)
                print(f"r={r:<4} {label(m):<32} vs. {label(best):<38} {fmt(p0, lo, hi)}")
                self.add("cost_rel_diff", f"{m} vs best baseline ({best}), r={r}", p0, lo, hi)
                cost_plot[r][m] = (p0, lo, hi)
                series.append((f"{label(m)} vs. {label(best)}", m, rel, p0))
            if r == r_hist:
                self.hist_panels.append((
                    f"Gesamtkosten bei Kostenverhältnis r={r}: relative Differenz",
                    "relative Kostendifferenz (negativ = Modell günstiger)", series))
        plots.plot_costs(cost_plot, self.fig_dir / "cost_sensitivity.png")

    def stability(self):
        """Lagerbestand bei 95 % Servicegrad je Split (Punktschaetzung, ohne Intervall)."""
        res, n, h = self.res, self.prep.n_series, self.st.horizon
        print(RULE)
        print("Stabilitaet ueber die Splits: Lagerbestand bei 95 % Servicegrad")
        stab = []
        for i, split in enumerate(res.splits, 1):
            fill = 1 - split.lost.sum(axis=1) / split.demand.sum()
            inventory = split.held.sum(axis=1) / (n * h)
            row = {"split": i}
            for pol in dict.fromkeys(res.policies):
                cols = np.where(res.policies == pol)[0]
                row[pol] = inventory_at_service_level(fill[cols], inventory[cols], 0.95)
            stab.append(row)
        stab_df = pd.DataFrame(stab)
        print(stab_df.rename(columns=plots.policy_label).round(2).to_string(index=False))
        plots.plot_stability(stab_df, self.fig_dir / "stability_splits.png")

    def feature_importance(self):
        st, res, feats = self.st, self.res, self.prep.feats
        qg = st.quantile_grid
        print(RULE)
        print("Merkmalswichtigkeit (Anteil an der Gesamtwichtigkeit, Mittel ueber die Splits)")

        def aggregate(per_split):
            out = {}
            for q in per_split[0]:
                arr = np.vstack([s[q] for s in per_split])
                out[q] = (np.array(feats), arr.mean(axis=0), arr.min(axis=0), arr.max(axis=0))
            return out

        imp_w = aggregate([s.imp_window for s in res.splits])
        imp_d = aggregate([s.imp_daily for s in res.splits])
        rows = []
        for model_name, imp in (("bedarfssumme", imp_w), ("taeglich", imp_d)):
            for q, (names, mean, lo, hi) in imp.items():
                for name, m_, l_, h_ in zip(names, mean, lo, hi):
                    rows.append({"model": model_name, "quantile": q, "feature": name,
                                 "importance_mean": m_, "importance_min": l_, "importance_max": h_})
        pd.DataFrame(rows).to_csv(self.out_dir / "feature_importance.csv", index=False)
        q_show = sorted({qg[0], min(qg, key=lambda x: abs(x - 0.9)),
                         min(qg, key=lambda x: abs(x - 0.99))})
        ref = min(q_show, key=lambda x: abs(x - 0.9))
        names, mean, _, _ = imp_w[ref]
        for i in np.argsort(mean)[::-1][:8]:
            print(f"  {plots.feature_label(names[i]):<45} {mean[i]:6.1%}")
        plots.plot_feature_importance(
            {q: imp_w[q] for q in q_show}, self.fig_dir / "feature_importance.png",
            title=f"Merkmalswichtigkeit: Bedarfssumme über {st.window} Tage")
        plots.plot_feature_importance(
            {q: imp_d[q] for q in st.daily_quantiles}, self.fig_dir / "feature_importance_daily.png",
            title="Merkmalswichtigkeit: tägliche Prognose")

    def run_info(self):
        st, prep, d = self.st, self.prep, self.st.cfg["data"]
        h = st.horizon
        info = {"stores": d["stores"], "categories": d["categories"], "n_series": int(prep.n_series),
                "horizon": h, "step_days": st.step, "n_splits": len(prep.cutoffs),
                "test_periods": [[str(prep.dates[c])[:10], str(prep.dates[c + h - 1])[:10]]
                                 for c in prep.cutoffs]}
        (self.out_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
