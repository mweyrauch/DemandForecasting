"""Fuehrt alle Auswertungslaeufe aus configs/confirmation.yaml aus und fasst sie zusammen.

Bereits vorhandene Laeufe (summary_ci.csv im Ausgabeordner) werden uebersprungen, ausser mit --force.
Jeder Lauf startet in einem eigenen Prozess, damit der Arbeitsspeicher danach wieder frei wird.

Beispiele:
    python scripts/run_confirmation.py                  # alle fehlenden Laeufe, dann Zusammenfassung
    python scripts/run_confirmation.py --only CA_2 TX_1  # nur bestimmte Laeufe (nach Namen)
    python scripts/run_confirmation.py --force           # alles neu rechnen
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dfi.summary import build_command, load_run_specs, summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml", help="Basis-Konfiguration")
    ap.add_argument("--runs", default="configs/confirmation.yaml", help="Lauf-Definitionen")
    ap.add_argument("--only", nargs="+", help="nur Laeufe, deren Name einen dieser Texte enthaelt")
    ap.add_argument("--force", action="store_true", help="auch vorhandene Laeufe neu rechnen")
    ap.add_argument("--out", default="results/confirmation")
    args = ap.parse_args()

    specs = load_run_specs(ROOT / args.runs)
    if args.only:
        specs = [s for s in specs if any(o.lower() in s["name"].lower() for o in args.only)]
    if not specs:
        sys.exit("Keine passenden Laeufe gefunden.")

    for i, spec in enumerate(specs, 1):
        done = (ROOT / spec["out_dir"] / "summary_ci.csv").exists()
        if done and not args.force:
            print(f"[{i}/{len(specs)}] {spec['name']}: vorhanden, uebersprungen ({spec['out_dir']})")
            continue
        print(f"[{i}/{len(specs)}] {spec['name']}: starte Lauf ...", flush=True)
        cmd = build_command(spec, args.config, sys.executable, str(ROOT / "scripts/train_forecast.py"))
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"Lauf '{spec['name']}' ist fehlgeschlagen (Rueckgabewert {result.returncode}). "
                     "Bereits fertige Laeufe bleiben erhalten; nach der Korrektur erneut starten.")

    all_dirs = [s["out_dir"] for s in load_run_specs(ROOT / args.runs)]
    ready = [d for d in all_dirs if (ROOT / d / "summary_ci.csv").exists()]
    df = summarize([ROOT / d for d in ready], ROOT / args.out)
    print("\nZusammenfassung:")
    print(df.assign(ergebnis=lambda d: d.apply(
        lambda r: f"{r.estimate:+.1%} [{r.ci_low:+.1%}, {r.ci_high:+.1%}]", axis=1)
    )[["run", "metric", "ergebnis"]].to_string(index=False))
    print(f"\nGespeichert in {args.out}/")


if __name__ == "__main__":
    main()
