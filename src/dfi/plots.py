"""Abbildungen im einheitlichen Pastell-Stil (Farben, Marker, Linienstile, Schraffuren).

Jede Politik hat in allen Plots dieselbe Farbe, denselben Marker, Linienstil und dieselbe
Schraffur (Zuordnung ueber ``set_policies``), damit sie auch in Graustufen unterscheidbar bleibt.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

PALETTE = ["#9EC5E8", "#C3B1E1", "#F4B6C2", "#FFC9A3", "#A8D8B9", "#F7D794", "#B5EAEA", "#D9C2A7"]
MARKERS = ["o", "D", "s", "P", "^", "v", "X", "h"]
LINESTYLES = ["-", "--", "-.", ":", "--", "-.", ":", "-"]
HATCHES = ["xx", "\\\\", "//", "++", "..", "oo", "--", "||"]

POLICY_ORDER: list[str] = []
POLICY_STYLE: dict[str, dict] = {}


def policy_label(name: str) -> str:
    floor = name.endswith("_floor")
    base = name[:-6] if floor else name
    if base == "baseline_mean_z":
        lab = "Mittelwert + z·σ"
    elif base.startswith("baseline_empirical_q"):
        d = base.split("_")[-1]
        lab = "Empirisches Quantil" + (f" ({d} T.)" if d.isdigit() else "")
    elif base == "quantile_model_window":
        lab = "Quantilmodell"
    else:
        lab = name
    return lab + (" + Mindestbestand" if floor else "")


def set_policies(names) -> None:
    """Legt Reihenfolge und Stil der Politiken fest (Modelle zuerst, dann Baselines)."""
    POLICY_ORDER[:] = list(names)
    POLICY_STYLE.clear()
    for i, n in enumerate(POLICY_ORDER):
        k = i % len(PALETTE)
        POLICY_STYLE[n] = dict(label=policy_label(n), color=PALETTE[k], marker=MARKERS[k],
                               ls=LINESTYLES[k], hatch=HATCHES[k])


set_policies(["quantile_model_window", "baseline_mean_z", "baseline_empirical_q"])

QUANTILE_STYLES = [
    dict(color="#9EC5E8", marker="o", ls="-", hatch="xx"),
    dict(color="#F7D794", marker="D", ls="--", hatch="//"),
    dict(color="#C3B1E1", marker="v", ls="-.", hatch="\\\\"),
    dict(color="#A8D8B9", marker="^", ls=":", hatch="++"),
]
EDGE_FACTOR = 0.55            # Kanten und Linien dunkler als die Fuellung
plt.rcParams["hatch.linewidth"] = 0.8


def darken(color: str, factor: float = EDGE_FACTOR):
    return tuple(c * factor for c in to_rgb(color))


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _style_axes(ax):
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _err(point, lo, hi):
    return max(0.0, point - lo), max(0.0, hi - point)


def _grouped_bars(ax, groups, series, pct=False):
    """groups: x-Beschriftungen; series: Liste (Stil, {gruppe: (wert, lo, hi)})."""
    k = len(series)
    width = 0.8 / k
    x = np.arange(len(groups))
    for j, (style, data) in enumerate(series):
        vals, errs, missing = [], [[], []], []
        for g in groups:
            v = data.get(g)
            if v is None or np.isnan(v[0]):
                vals.append(0.0)
                errs[0].append(0.0)
                errs[1].append(0.0)
                missing.append(True)
            else:
                vals.append(v[0])
                e = _err(*v) if not np.isnan(v[1]) else (0.0, 0.0)
                errs[0].append(e[0])
                errs[1].append(e[1])
                missing.append(False)
        xs = x + (j - (k - 1) / 2) * width
        ax.bar(xs, vals, width * 0.92, color=style["color"], edgecolor=darken(style["color"]),
               hatch=style.get("hatch"), linewidth=0.9, label=style["label"],
               yerr=errs, capsize=2, error_kw=dict(ecolor=darken(style["color"], 0.4), lw=1.0))
        for xi, m in zip(xs, missing):
            if m:
                ax.annotate("n. e.", (xi, 0), ha="center", va="bottom", fontsize=6, color="gray",
                            rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in groups])
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    _style_axes(ax)


def plot_tradeoff(curve, path, bands=None, fill_max=0.995):
    """Kompromisskurve Servicegrad gegen Lagerbestand.

    ``bands``: optional {Politik: (Servicegrad-Raster, untere Grenze, obere Grenze)} fuer das
    Bootstrap-Fehlerband (Lagerbestand-Intervall bei gegebenem Servicegrad).
    ``fill_max``: Punkte oberhalb dieses Servicegrads werden nicht gezeichnet (Saettigungsbereich,
    in dem alle Kurven ueberlappen und die x-Achse unnoetig streckt).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for pol in POLICY_ORDER:
        g = curve[(curve["policy"] == pol) & (curve["fill_rate"] <= fill_max)].sort_values("fill_rate")
        if g.empty:
            continue
        st = POLICY_STYLE[pol]
        if bands and pol in bands and len(bands[pol][0]) > 1:
            grid, lo, hi = (np.asarray(a) for a in bands[pol])
            keep = grid <= fill_max
            ax.fill_betweenx(grid[keep], lo[keep], hi[keep], color=st["color"], alpha=0.40,
                             linewidth=0, zorder=1)
        ax.plot(g["avg_inventory"], g["fill_rate"], color=darken(st["color"], 0.8),
                linestyle=st["ls"], marker=st["marker"], markersize=6, linewidth=1.8,
                markerfacecolor=st["color"], markeredgecolor=darken(st["color"]),
                label=st["label"], zorder=2)
    ax.set_xlabel("durchschnittlicher Lagerbestand (Einheiten je Artikel)")
    ax.set_ylabel("Servicegrad (Fill Rate)")
    title = "Servicegrad gegen Lagerbestand"
    if bands:
        title += "\n(Band: 95 %-Bootstrap-Intervall des Lagerbestands je Servicegrad)"
    ax.set_title(title, fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_inventory_at_sl(data, path):
    """data: {Servicegrad: {Politik: (Wert, lo, hi)}}. Balken mit 95 %-Intervall."""
    levels = list(data)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    series = [(POLICY_STYLE[p], {f"{sl:.0%}": data[sl][p] for sl in levels if p in data[sl]})
              for p in POLICY_ORDER]
    _grouped_bars(ax, [f"{sl:.0%}" for sl in levels], series)
    ax.set_xlabel("Ziel-Servicegrad")
    ax.set_ylabel("durchschnittlicher Lagerbestand (Einheiten je Artikel)")
    ax.set_title("Lagerbestand bei festem Servicegrad (95 %-Intervall; n. e. = nicht erreicht)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, path)


def plot_pinball(series, path):
    """series: Liste (Politik-Schluessel, Label, [(q, Verbesserung, lo, hi), ...])."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    qs = [f"{q:g}" for q, *_ in series[0][2]]
    bars = []
    for key, label, rows in series:
        bars.append((dict(POLICY_STYLE[key], label=label),
                     {f"{q:g}": (p, lo, hi) for q, p, lo, hi in rows}))
    _grouped_bars(ax, qs, bars, pct=True)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Quantil q")
    ax.set_ylabel("Verbesserung des Pinball-Loss")
    ax.set_title("Bedarfssumme: Pinball-Loss relativ zur besten empirischen Baseline\n"
                 "(positiv = Modell besser)", fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_costs(data, path):
    """data: {Kostenverhaeltnis: {Politik: (Wert, lo, hi)}}; relative Kostendifferenz zur besten Baseline."""
    ratios = list(data)
    keys = list(data[ratios[0]])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    series = [(dict(POLICY_STYLE[k], label=f"{policy_label(k)} vs. beste Baseline"),
               {str(r): data[r][k] for r in ratios if k in data[r]}) for k in keys]
    _grouped_bars(ax, [str(r) for r in ratios], series, pct=True)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Kostenverhältnis r (Fehlmenge / Lagerkosten je Einheit und Tag)")
    ax.set_ylabel("relative Kostendifferenz")
    ax.set_title("Minimale Gesamtkosten (negativ = Modell günstiger)")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_stability(df, path, level_label="95 %"):
    """df: Spalte 'split' und je Politik eine Spalte mit dem Lagerbestand."""
    splits = [str(s) for s in df["split"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    series = []
    for p in POLICY_ORDER:
        if p in df.columns:
            series.append((POLICY_STYLE[p], {str(s): (v, np.nan, np.nan)
                                             for s, v in zip(df["split"], df[p])}))
    _grouped_bars(ax, splits, series)
    ax.set_xlabel("Split (rollierender Cutoff)")
    ax.set_ylabel("Lagerbestand (Einheiten je Artikel)")
    ax.set_title(f"Stabilität über die Splits: Lagerbestand bei {level_label} Servicegrad")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, path)


def plot_wape(data, path):
    """data: {Name: (WAPE, lo, hi)}; zwei Balken mit Intervall."""
    names = list(data)
    styles = [QUANTILE_STYLES[1], QUANTILE_STYLES[0]]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    series = [(dict(styles[i % 2], label=n), {"WAPE": data[n]}) for i, n in enumerate(names)]
    _grouped_bars(ax, ["WAPE"], series)
    ax.set_ylabel("WAPE (kleiner ist besser)")
    ax.set_title("Tägliche Prognose")
    ax.legend()
    _save(fig, path)


def plot_bootstrap_histograms(panels, path):
    """panels: Liste (Titel, x-Beschriftung, [(Label, Politik-Schluessel, Ziehungen, Punktschaetzung)]).

    Echte Histogramme der Bootstrap-Verteilung. Vertikale Linien: Nulllinie (grau, gestrichelt),
    Punktschaetzung (durchgezogen mit Marker), 95 %-Intervall (gepunktet).
    """
    k = len(panels)
    cols = 2
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4 * rows), squeeze=False)
    for ax, (title, xlabel, series) in zip(axes.ravel(), panels):
        allv = np.concatenate([np.asarray(d)[~np.isnan(d)] for _, _, d, _ in series])
        bins = np.linspace(allv.min(), allv.max(), 26)
        top = 0.0
        for label, key, draws, point in series:
            st = POLICY_STYLE[key]
            d = np.asarray(draws, float)
            d = d[~np.isnan(d)]
            n_, _, _ = ax.hist(d, bins=bins, color=st["color"], edgecolor=darken(st["color"]),
                               hatch=st["hatch"], alpha=0.8, label=label)
            top = max(top, n_.max())
        for label, key, draws, point in series:
            st = POLICY_STYLE[key]
            d = np.asarray(draws, float)
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            c = darken(st["color"], 0.7)
            ax.axvline(point, color=c, linestyle="-", linewidth=2)
            ax.plot([point], [top * 1.04], marker=st["marker"], markersize=8,
                    markerfacecolor=st["color"], markeredgecolor=c, linestyle="none")
            ax.axvline(lo, color=c, linestyle=":", linewidth=1.5)
            ax.axvline(hi, color=c, linestyle=":", linewidth=1.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        ax.set_ylabel("Anzahl Bootstrap-Ziehungen")
        ax.set_ylim(0, top * 1.15)
        ax.legend(fontsize=8)
        _style_axes(ax)
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    fig.suptitle("Bootstrap-Verteilungen (durchgezogen: Punktschätzung, gepunktet: 95 %-Intervall)",
                 fontsize=11)
    _save(fig, path)


def feature_label(name: str) -> str:
    """Lesbare deutsche Beschriftung der Merkmale."""
    fixed = {"sell_price": "Verkaufspreis", "price_rel": "Preis relativ zum Artikelmittel",
             "has_event": "Ereignis (Feiertag o. ä.)", "snap": "SNAP-Tag", "wday": "Wochentag",
             "month": "Monat", "dayofmonth": "Tag im Monat", "item_code": "Artikel (Code)",
             "store_code": "Filiale (Code)", "dept_code": "Abteilung (Code)"}
    if name in fixed:
        return fixed[name]
    parts = name.split("_")
    if parts[0] == "lag" and len(parts) == 2:
        return f"Verkäufe vor {parts[1]} Tagen"
    if parts[0] == "rmean" and len(parts) == 3:
        return f"Ø {parts[2]} Tage, {parts[1]} Tage verzögert"
    if parts[0] == "rzero" and len(parts) == 3:
        return f"Anteil Tage ohne Verkauf ({parts[2]} Tage)"
    if parts[0] == "rstd" and len(parts) == 3:
        return f"Streuung der Verkäufe ({parts[2]} Tage)"
    return name


def plot_feature_importance(data, path, top_n: int = 10, title=None):
    """data: {q: (Merkmale, Mittel, Minimum, Maximum)} ueber die Splits.

    Waagerechte, gruppierte Balken (Anteil an der Gesamtwichtigkeit); die Fehlerbalken zeigen die
    Spannweite ueber die rollierenden Splits.
    """
    qs = sorted(data)
    feats = list(data[qs[0]][0])
    mean_all = np.mean([data[q][1] for q in qs], axis=0)
    order = np.argsort(mean_all)[::-1][:top_n][::-1]        # wichtigste oben
    k = len(qs)
    height = 0.8 / k
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 1.6 + 0.5 * len(order)))
    for j, q in enumerate(qs):
        st = QUANTILE_STYLES[j % len(QUANTILE_STYLES)]
        _, mean, lo, hi = data[q]
        mean, lo, hi = mean[order], lo[order], hi[order]
        ys = y + (j - (k - 1) / 2) * height
        ax.barh(ys, mean, height * 0.92, color=st["color"], edgecolor=darken(st["color"]),
                hatch=st["hatch"], linewidth=0.9, label=f"q = {q:g}",
                xerr=[np.maximum(mean - lo, 0), np.maximum(hi - mean, 0)], capsize=3,
                error_kw=dict(ecolor=darken(st["color"], 0.4), lw=1.1))
        ax.plot(mean, ys, linestyle="none", marker=st["marker"], markersize=5,
                markerfacecolor=st["color"], markeredgecolor=darken(st["color"]), zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([feature_label(feats[i]) for i in order])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Anteil an der Gesamtwichtigkeit")
    ax.set_title((title or "Merkmalswichtigkeit der Quantilmodelle")
                 + "\n(Mittel über die Splits; Fehlerbalken: Spannweite)", fontsize=10)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_forest(panels, path):
    """Vergleich mehrerer Laeufe: je Kennzahl ein Panel mit Punktschaetzung und 95 %-Intervall.

    panels: Liste (Titel, x-Beschriftung, [(Lauf, Schaetzung, lo, hi), ...]).
    Jeder Lauf hat eigene Farbe, Marker und Linienart (gleich in allen Panels).
    """
    runs = list(dict.fromkeys(r for _, _, rows in panels for r, *_ in rows))
    style = {r: dict(color=PALETTE[i % len(PALETTE)], marker=MARKERS[i % len(MARKERS)],
                     ls=LINESTYLES[i % len(LINESTYLES)]) for i, r in enumerate(runs)}
    k = len(panels)
    fig, axes = plt.subplots(1, k, figsize=(4.6 * k + 2, 0.55 * len(runs) + 2.4), squeeze=False)
    for ax, (title, xlabel, rows) in zip(axes[0], panels):
        for yi, (run, est, lo, hi) in enumerate(rows):
            st = style[run]
            c = darken(st["color"], 0.7)
            ax.hlines(yi, lo, hi, color=c, linestyles=st["ls"], linewidth=2.2)
            ax.plot([est], [yi], marker=st["marker"], markersize=9, linestyle="none",
                    markerfacecolor=st["color"], markeredgecolor=c, zorder=3)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.2)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r for r, *_ in rows] if ax is axes[0][0] else [], fontsize=8)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Bestätigung: Modell gegen die beste Baseline in mehreren Läufen (95 %-Intervall)",
                 fontsize=11)
    _save(fig, path)
