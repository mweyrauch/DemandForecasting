"""Zusammenfassung mehrerer Auswertungslaeufe (Bestaetigung auf anderen Filialen/Zeitraeumen)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from . import plots

# (Anzeigename, Kennzahl, Praefix des Vergleichs, Suffix des Vergleichs, x-Beschriftung)
KEY_METRICS = [
    ("Lagerbestand bei 95 % Servicegrad", "inventory_rel_diff", "quantile_model_window vs best baseline", "SL=0.95",
     "relative Differenz des Lagerbestands (negativ = Modell besser)"),
    ("Gesamtkosten bei r=20", "cost_rel_diff", "quantile_model_window vs best baseline", "r=20",
     "relative Kostendifferenz (negativ = Modell besser)"),
    ("Pinball-Loss bei q=0,9", "pinball_improvement", "quantile_model_window vs best empirical", "q=0.9",
     "Verbesserung des Pinball-Loss (positiv = Modell besser)"),
]


def run_label(run_dir: Path) -> str:
    """Beschriftung eines Laufs aus ``run_info.json`` (Filiale, Zeitraum) oder dem Ordnernamen."""
    info_file = Path(run_dir) / "run_info.json"
    if info_file.exists():
        info = json.loads(info_file.read_text())
        stores = ",".join(info.get("stores") or ["alle"])
        periods = info.get("test_periods") or []
        span = f"{periods[0][0][:7]} bis {periods[-1][1][:7]}" if periods else ""
        return f"{stores}, {span}".strip(", ")
    return Path(run_dir).name


def collect_key_metrics(run_dirs) -> pd.DataFrame:
    """Liest ``summary_ci.csv`` je Lauf und gibt die Schluesselkennzahlen als Tabelle zurueck."""
    rows = []
    for rd in run_dirs:
        rd = Path(rd)
        df = pd.read_csv(rd / "summary_ci.csv")
        label = run_label(rd)
        for title, metric, prefix, suffix, _ in KEY_METRICS:
            sub = df[(df["metric"] == metric) & df["comparison"].str.startswith(prefix + " ")
                     & df["comparison"].str.endswith(suffix)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append({"run": label, "dir": str(rd), "metric": title, "estimate": r["estimate"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows)


# Zuordnung Konfigurationsschluessel -> Kommandozeilenargument von train_forecast.py
ARG_FLAGS = {"stores": "--stores", "categories": "--categories", "max_series": "--max-series",
             "n_splits": "--n-splits", "step_days": "--step-days"}


def load_run_specs(path) -> list[dict]:
    """Liest die Lauf-Definitionen (Name, Ausgabeordner, Argumente) aus einer YAML-Datei."""
    return yaml.safe_load(Path(path).read_text())["runs"]


def build_command(spec: dict, base_config: str, python: str, script: str) -> list[str]:
    """Kommandozeile fuer train_forecast.py zu einer Lauf-Definition."""
    cmd = [python, script, "--config", base_config, "--out-dir", spec["out_dir"]]
    for key, value in (spec.get("args") or {}).items():
        flag = ARG_FLAGS[key]
        cmd += [flag] + [str(v) for v in value] if isinstance(value, list) else [flag, str(value)]
    return cmd


def missing_runs(run_dirs, root=".") -> list[str]:
    """Ausgabeordner, in denen noch keine ``summary_ci.csv`` liegt (Lauf fehlt oder ist abgebrochen)."""
    return [str(d) for d in run_dirs if not (Path(root) / d / "summary_ci.csv").exists()]


def summarize(run_dirs, out_dir) -> pd.DataFrame:
    """Schreibt Tabelle und Forest-Plot fuer die angegebenen Laeufe und gibt die Tabelle zurueck."""
    df = collect_key_metrics(run_dirs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "confirmation_summary.csv", index=False)
    panels = []
    for title, _, _, _, xlabel in KEY_METRICS:
        sub = df[df["metric"] == title]
        if not sub.empty:
            panels.append((title, xlabel, [(r.run, r.estimate, r.ci_low, r.ci_high)
                                           for r in sub.itertuples()]))
    if panels:
        plots.plot_forest(panels, out / "confirmation_forest.png")
    return df
