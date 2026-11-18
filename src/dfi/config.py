"""Konfiguration laden: YAML-Datei, optional auf einer Basisdatei aufbauend."""
from __future__ import annotations

from pathlib import Path

import yaml

# Kommandozeilen-Argument -> (Abschnitt, Schluessel) in der Konfiguration
OVERRIDES = {"stores": ("data", "stores"), "categories": ("data", "categories"),
             "max_series": ("data", "max_series"), "n_splits": ("evaluation", "n_splits"),
             "step_days": ("evaluation", "step_days")}


def deep_merge(base: dict, update: dict) -> dict:
    """Legt ``update`` ueber ``base``. Verschachtelte Abschnitte werden zusammengefuehrt, alle
    anderen Werte (auch Listen) ersetzt."""
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path, **overrides) -> dict:
    """Liest eine YAML-Konfiguration.

    Enthaelt die Datei ``extends: <Datei>`` (relativ zu ihr selbst), wird zuerst diese Basis
    geladen; die Datei muss dann nur die abweichenden Werte nennen. Gesetzte ``overrides``
    (siehe ``OVERRIDES``, z. B. von der Kommandozeile) ersetzen einzelne Werte.
    """
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    for name, value in overrides.items():
        if value:
            section, key = OVERRIDES[name]
            cfg[section][key] = value
    return cfg
