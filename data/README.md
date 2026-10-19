# Daten

Die M5-Rohdaten werden **nicht** in dieses Repository eingecheckt.

## Variante A: kagglehub (Standard)

```bash
pip install -e ".[kaggle]"
```

1. Kaggle-Zugangsdaten bereitstellen (z. B. `KAGGLE_USERNAME` und `KAGGLE_KEY` als
   Umgebungsvariablen oder `kagglehub.login()`).
2. Die Regeln des Wettbewerbs auf Kaggle lesen und akzeptieren.
3. Das Skript laedt die Daten beim ersten Lauf automatisch in den kagglehub-Cache
   (ausserhalb des Repositories). Einstellung in `configs/default.yaml`:

```yaml
data:
  source: kagglehub
  kagglehub:
    kind: competition
    handle: m5-forecasting-accuracy
```


## Variante B: lokale Dateien

`source: local` setzen und diese Dateien in `data/` ablegen: `calendar.csv`,
`sales_train_evaluation.csv`, `sell_prices.csv`.

## Ohne echte Daten (Test)

```bash
python scripts/make_synthetic_m5.py --out data/synthetic
python scripts/train_forecast.py --config configs/synthetic.yaml
```
