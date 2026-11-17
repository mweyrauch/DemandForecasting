"""Rollierende Auswertung: Quantil-Modelle gegen Baselines, mit Bootstrap-Konfidenzintervallen.

Fuer jeden Cutoff (``evaluation.n_splits``, je ``horizon`` Tage Test) werden die Quantil-Modelle
trainiert und alle Politiken simuliert (``dfi.pipeline``). Danach werden die Splits zusammengefasst
und ausgewertet (``dfi.report``).

Beispiele:
    python scripts/train_forecast.py
    python scripts/train_forecast.py --stores CA_2 --out-dir results/confirm_CA_2
"""
from __future__ import annotations

import argparse

from dfi.pipeline import Settings, load_config, pool, prepare, run_split
from dfi.report import Report


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stores", nargs="+", help="Filialen (ueberschreibt die Konfiguration)")
    ap.add_argument("--categories", nargs="+", help="Kategorien (ueberschreibt die Konfiguration)")
    ap.add_argument("--max-series", type=int, help="maximale Artikelzahl (ueberschreibt die Konfiguration)")
    ap.add_argument("--n-splits", type=int, help="Anzahl rollierender Cutoffs")
    ap.add_argument("--step-days", type=int, help="Abstand der Cutoffs in Tagen (Standard: horizon)")
    ap.add_argument("--out-dir", default="results", help="Ausgabeordner fuer Tabellen und Abbildungen")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, stores=args.stores, categories=args.categories,
                      max_series=args.max_series, n_splits=args.n_splits, step_days=args.step_days)
    st = Settings.from_config(cfg)
    prep = prepare(cfg, st)

    splits = []
    for i, c in enumerate(prep.cutoffs, 1):
        first, last = str(prep.dates[c])[:10], str(prep.dates[c + st.horizon - 1])[:10]
        print(f"\n--- Split {i}/{len(prep.cutoffs)}: Testfenster {first} bis {last} ---", flush=True)
        splits.append(run_split(prep, st, c))

    Report(st, prep, pool(splits), args.out_dir).run()


if __name__ == "__main__":
    main()
