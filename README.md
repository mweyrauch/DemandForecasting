# Nachfrageprognose und Bestandsoptimierung (M5)

> **Status: in Arbeit.** Pipeline und Auswertung laufen, die Ergebnisse unten sind noch auszufuellen.

## Fragestellung

Wie stark verbessern probabilistische Nachfrageprognosen (Quantil-Modelle) Servicegrad und
Lagerbestand gegenueber einfachen Durchschnittsprognosen?

## Kernergebnis

_TODO: ein bis zwei Saetze mit konkreten Zahlen aus deiner Auswertung, z. B.
"Bei gleichem Servicegrad von 95 % sinkt der durchschnittliche Lagerbestand um X %."_

![Servicegrad vs. Lagerbestand](results/figures/inventory_tradeoff.png)

## Daten

[M5 Forecasting (Walmart)](https://www.kaggle.com/c/m5-forecasting-accuracy), taegliche
Verkaufszahlen mit Preisen und Kalender. Die Daten sind **nicht** Teil des Repositories und werden per `kagglehub` geladen
(siehe [`data/README.md`](data/README.md)). Nutzungsbedingungen des Wettbewerbs beachten.

## Wichtige Kennzahlen

| Begriff | Bedeutung |
|---|---|
| **Lieferzeit** | Tage zwischen Bestellung und Wareneingang (Annahme: 3 Tage). |
| **Bedarfssumme** | Summe der Nachfrage ueber *Lieferzeit + 1* Tage ab heute. Der Bestand muss reichen, bis die naechste Bestellung eintrifft. Das Quantilmodell sagt genau diese Groesse voraus. |
| **Zielbestand (Order-up-to-Niveau)** | Soll-Bestand inklusive bereits bestellter Ware. Bestellmenge = Zielbestand minus (Lagerbestand + Ware unterwegs). Der Zielbestand ist ein Quantil der Bedarfssumme. |
| **Quantil q** | "Mit Wahrscheinlichkeit q ist die Bedarfssumme hoechstens so gross." Ein hoeheres q bedeutet mehr Sicherheit und mehr Lager. |
| **Kalibrierung** | Anteil der Faelle, in denen die Nachfrage tatsaechlich unter dem vorhergesagten Quantil lag. Sollte ungefaehr q entsprechen. |
| **Servicegrad (Fill Rate)** | 1 minus entgangene Menge geteilt durch Gesamtnachfrage, mengengewichtet ueber alle Artikel. Nicht zu verwechseln mit q: q ist die Wahrscheinlichkeit, keine Fehlmenge zu haben, der Servicegrad der Anteil gelieferter Einheiten. |
| **Durchschnittlicher Lagerbestand** | Mittlerer Bestand am Tagesende in Einheiten je Artikel. Naeherung fuer die Kapitalbindung. |
| **Fehlmenge (Lost Sales)** | Nachfrage, die nicht bedient werden konnte. Sie geht verloren und wird nicht nachgeliefert. |
| **Kostenverhaeltnis r** | Kosten einer fehlenden Einheit relativ zu den Lagerkosten einer Einheit pro Tag. Gesamtkosten = Lager-Einheitentage + r mal Fehlmenge. |
| **WAPE** | Summe der absoluten Fehler geteilt durch die Summe der tatsaechlichen Verkaeufe. |
| **MASE** | Fehler im Verhaeltnis zum Fehler einer naiven Prognose (unter 1 ist besser als naiv). |
| **Pinball-Loss** | Verlustfunktion fuer Quantilprognosen: bestraft zu niedrige und zu hohe Werte asymmetrisch je nach q. |

## Methode

1. **Baselines:** Seasonal Naive und gleitender Durchschnitt (`dfi/baselines.py`).
2. **Features ohne Leakage:** Alle Lags und Rolling-Statistiken sind um mindestens den
   Prognosehorizont (28 Tage) verschoben (`dfi/features.py`). Dadurch ist keine rekursive
   Prognose noetig, und jedes Feature ist zum Prognosezeitpunkt bekannt.
3. **Quantil-Modelle:** Gradient Boosting (LightGBM) fuer mehrere Quantile, nicht kreuzend
   (`dfi/models.py`).
4. **Validierung:** zeitlich getrennt, rollierende Ursprungszeitpunkte; Metriken WAPE, MASE,
   Pinball-Loss und Kalibrierung (`dfi/evaluate.py`).
5. **Bestandssimulation:** Order-up-to-Politik mit Lieferzeit (`dfi/inventory.py`). Drei Politiken
   werden ueber je einen Parameterbereich verglichen:
   - `baseline_mean_z`: Mittelwert der letzten 28 Tage plus z mal Standardabweichung
   - `baseline_empirical_q`: empirisches Quantil der Bedarfssummen der letzten 12 Wochen
     (starke Baseline, beruecksichtigt sporadische Nachfrage)
   - `quantile_model_window`: Quantilmodell, das die Bedarfssumme ueber Lieferzeit + 1 Tag
     direkt vorhersagt. Das Trainingsfenster endet so, dass kein Zielfenster in den Testzeitraum
     reicht (kein Leakage).
6. **Fairer Vergleich:** Statt einzelner Punkte werden (a) die Kompromisskurven, (b) der
   Lagerbestand bei festem Servicegrad (lineare Interpolation, keine Extrapolation) und
   (c) die minimalen Gesamtkosten je Kostenverhaeltnis verglichen. Jede Politik darf ihren
   Parameter fuer jedes Ziel neu waehlen.

## Ergebnisse

_TODO: Tabelle (Modell, WAPE, MASE, Servicegrad, Lagerbestand) und Interpretation._

## Annahmen und Grenzen

- Lieferzeit und Bestellpolitik sind **vereinfachte Annahmen**. Statt fester Kostensaetze wird ein Bereich von Kostenverhaeltnissen betrachtet.
- Nicht gedeckte Nachfrage geht verloren (Lost Sales); keine Mindestbestellmengen.
- Die Simulation zeigt Tendenzen unter diesen Annahmen, keine garantierten Einsparungen.
- Teilmenge der Daten (siehe `configs/default.yaml`), nicht der vollstaendige Datensatz.

## Reproduzieren

```bash
pip install -e ".[dev,kaggle]"
# Kaggle-Zugangsdaten setzen und Wettbewerbsregeln akzeptieren (siehe data/README.md)
python scripts/run_baselines.py
python scripts/train_forecast.py
pytest
```

Schneller Test ohne echte Daten:

```bash
python scripts/make_synthetic_m5.py --out data/synthetic
python scripts/train_forecast.py --config configs/synthetic.yaml
```
