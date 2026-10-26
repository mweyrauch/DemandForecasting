"""Fasst mehrere Auswertungslaeufe zusammen (Tabelle und Forest-Plot).

Ohne Argumente werden die Ausgabeordner aus configs/confirmation.yaml verwendet.
Fehlende Laeufe lassen sich automatisch erzeugen mit:  python scripts/run_confirmation.py

Beispiel:
    python scripts/summarize_runs.py results results/confirm_CA_2 results/confirm_TX_1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dfi.summary import load_run_specs, missing_runs, summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", help="Ausgabeordner der Laeufe (mit summary_ci.csv)")
    ap.add_argument("--runs", default=str(ROOT / "configs/confirmation.yaml"),
                    help="Lauf-Definitionen, falls keine Ordner angegeben sind")
    ap.add_argument("--out", default="results/confirmation")
    ap.add_argument("--skip-missing", action="store_true", help="fehlende Laeufe ueberspringen")
    args = ap.parse_args()

    run_dirs = args.run_dirs or [s["out_dir"] for s in load_run_specs(args.runs)]
    missing = missing_runs(run_dirs)
    if missing:
        print("Es fehlt die Datei summary_ci.csv in:", *missing, sep="\n  ")
        print("Diese Dateien entstehen am Ende eines vollstaendigen Laufs von train_forecast.py.")
        print("Alle fehlenden Laeufe erzeugen:  python scripts/run_confirmation.py")
        if not args.skip_missing:
            sys.exit(1)
        run_dirs = [d for d in run_dirs if d not in missing]
    if not run_dirs:
        sys.exit("Keine Laeufe zum Zusammenfassen.")

    df = summarize(run_dirs, args.out)
    print(df.assign(ergebnis=lambda d: d.apply(
        lambda r: f"{r.estimate:+.1%} [{r.ci_low:+.1%}, {r.ci_high:+.1%}]", axis=1)
    )[["run", "metric", "ergebnis"]].to_string(index=False))
    print(f"\nGespeichert in {args.out}/")


if __name__ == "__main__":
    main()
