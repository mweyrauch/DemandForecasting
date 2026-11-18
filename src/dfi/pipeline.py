"""Ablauf der rollierenden Auswertung: Daten vorbereiten, je Split trainieren und simulieren.

Die drei Schritte:
  * ``prepare``    laedt die Daten und baut die Merkmale,
  * ``run_split``  trainiert die Quantil-Modelle fuer einen Cutoff und simuliert alle Politiken,
  * ``pool``       fasst die Splits zusammen (Grundlage fuer ``dfi.report``).
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import moving_average
from .data import load_m5, resolve_data_dir, sales_matrix, to_long
from .evaluate import rolling_origin_cutoffs
from .features import add_window_target, build_features, required_history, window_train_mask
from .inventory import empirical_quantile_targets, policy_name, simulate_order_up_to, window_sum
from .models import feature_importance_all, fit_quantile_models, predict_quantiles

MODEL = "quantile_model_window"
MODEL_FLOOR = "quantile_model_window_floor"

def pinball_elements(y, pred, q):
    d = y - pred
    return np.maximum(q * d, (q - 1) * d)


@dataclass
class Settings:
    """Aus der Konfiguration abgeleitete Groessen, die mehrere Schritte brauchen."""
    cfg: dict
    horizon: int
    lead_time: int
    window: int                 # Schutzzeitraum: Lieferzeit + 1 Tag
    daily_quantiles: list
    quantile_grid: list
    n_splits: int
    step: int
    min_stock: float
    baseline_specs: list
    baseline_names: list
    model_names: list
    lookbacks: list             # Rueckblicke fuer die Diagnose "Artikel ohne Verkauf"

    @classmethod
    def from_config(cls, cfg: dict) -> "Settings":
        f, inv, ev = cfg["forecast"], cfg["inventory"], cfg["evaluation"]
        specs = inv["baselines"]
        return cls(
            cfg=cfg, horizon=f["horizon"], lead_time=inv["lead_time"], window=inv["lead_time"] + 1,
            daily_quantiles=f["quantiles"], quantile_grid=inv["quantile_grid"],
            n_splits=ev["n_splits"], step=int(ev.get("step_days") or f["horizon"]),
            min_stock=float(inv.get("min_stock", 1)), baseline_specs=specs,
            baseline_names=[policy_name(s) for s in specs],
            model_names=[MODEL] + ([MODEL_FLOOR] if inv.get("model_floor") else []),
            lookbacks=sorted({28} | {int(s["days"]) for s in specs if s["type"] == "empirical"}),
        )

    @property
    def policy_names(self) -> list:
        return self.model_names + self.baseline_names


@dataclass
class Prepared:
    """Geladene Daten und Merkmale fuer alle Splits."""
    mat: np.ndarray             # Verkaeufe [Artikel x Tage]
    long: pd.DataFrame          # Long-Format mit Merkmalen, sortiert nach Artikel und Datum
    feats: list
    dates: np.ndarray           # Datum je Spalte von ``mat``
    cutoffs: list               # erster Testtag je Split (Spaltenindex)

    @property
    def n_series(self) -> int:
        return self.mat.shape[0]


@dataclass
class SplitResult:
    """Kennzahlen eines Splits; Arrays sind je Artikel, damit der Bootstrap sie gewichten kann."""
    err_ma: np.ndarray          # absolute Fehler des gleitenden Durchschnitts
    err_model: np.ndarray       # absolute Fehler des taeglichen Modells (q50)
    actual: np.ndarray          # Summe der Verkaeufe
    daily_hits: dict            # q -> Anzahl Beobachtungen <= Prognose
    daily_pinball: dict         # q -> Summe Pinball-Loss
    daily_total: int
    pinball: dict               # Politik -> {q: Summe Pinball-Loss der Bedarfssumme}
    calib_hits: dict            # q -> Anzahl Bedarfssummen <= Modellprognose
    calib_total: int
    points: list                # (Politik, Parameter) je Kurvenpunkt
    lost: np.ndarray            # entgangene Einheiten [Punkte x Artikel]
    held: np.ndarray            # Lager-Einheitentage [Punkte x Artikel]
    demand: np.ndarray          # Nachfrage im Testfenster
    imp_daily: dict             # q -> Merkmalswichtigkeit
    imp_window: dict
    zero_share: dict            # Rueckblick -> Anteil Artikel ohne Verkauf
    zero_demand: dict           # Rueckblick -> Testnachfrage dieser Artikel
    total_demand: float


@dataclass
class Pooled:
    """Ueber alle Splits zusammengefasste Kennzahlen."""
    err_ma: np.ndarray
    err_model: np.ndarray
    actual: np.ndarray
    daily_hits: dict
    daily_pinball: dict
    daily_total: int
    pinball: dict
    calib_hits: dict
    calib_total: int
    policies: np.ndarray
    params: np.ndarray
    lost: np.ndarray
    held: np.ndarray
    demand: np.ndarray
    zero_share: dict
    zero_demand: dict
    total_demand: float
    splits: list = field(repr=False, default_factory=list)   # die einzelnen SplitResults

    @property
    def n_splits(self) -> int:
        return len(self.splits)


def prepare(cfg: dict, st: Settings) -> Prepared:
    """Laedt die Daten, baut die Merkmale und bestimmt die Cutoffs."""
    d, f = cfg["data"], cfg["forecast"]
    sales, calendar, prices = load_m5(resolve_data_dir(d), d["sales_file"], d["stores"],
                                      d["categories"], d["max_series"], d["seed"])
    _, mat, day_cols = sales_matrix(sales)
    n, n_days = mat.shape

    # Historie: aeltester Cutoff - Trainingsfenster - Vorlauf fuer Lags und Rollfenster
    long_windows = tuple(f.get("long_windows", (91, 182, 364)))
    history = (f["train_days"] + required_history(st.horizon, long_windows) + st.horizon
               + (st.n_splits - 1) * st.step)
    long = to_long(sales, calendar, prices, last_days=history)
    print(f"Serien: {n}, Zeilen im Long-Format: {len(long):,}, Splits: {st.n_splits}")
    long, feats = build_features(long, horizon=st.horizon, long_windows=long_windows)
    long = add_window_target(long, st.lead_time)
    # Speicher sparen: nur noch benoetigte Spalten behalten (Zeilen bleiben nach Artikel und Datum
    # sortiert) und alle Merkmale als float32 ablegen.
    long = long[["date", "sales", "target_window"] + feats]
    for col in feats:
        long[col] = long[col].astype("float32")
    gc.collect()
    dates = pd.to_datetime(calendar.set_index("d").loc[day_cols, "date"]).to_numpy()
    cutoffs = rolling_origin_cutoffs(n_days, st.horizon, st.n_splits, st.step)
    return Prepared(mat=mat, long=long, feats=feats, dates=dates, cutoffs=cutoffs)


def policy_targets(st: Settings, train_mat, ma, model_pred) -> list:
    """Zielniveaus aller Politiken: (Name, Parameter, Ziel [Artikel x Tage], ist_quantil)."""
    inv = st.cfg["inventory"]
    n, h, w = train_mat.shape[0], st.horizon, st.window
    sigma = train_mat[:, -28:].std(axis=1)
    targets = []
    for q in st.quantile_grid:
        targets.append((MODEL, q, model_pred[q], True))
        if MODEL_FLOOR in st.model_names:
            targets.append((MODEL_FLOOR, q, np.maximum(model_pred[q], st.min_stock), True))
    for spec, name in zip(st.baseline_specs, st.baseline_names):
        floor = st.min_stock if spec.get("floor") else 0.0
        if spec["type"] == "mean_z":
            for z in inv["z_grid"]:
                t_ = np.array([window_sum(ma[i], w) + z * sigma[i] * np.sqrt(w) for i in range(n)])
                targets.append((name, z, np.maximum(t_, floor), False))
        else:
            for q in st.quantile_grid:
                emp = empirical_quantile_targets(train_mat, w, q, h, int(spec["days"]))
                targets.append((name, q, np.maximum(emp, floor), True))
    return targets


def run_split(prep: Prepared, st: Settings, c: int) -> SplitResult:
    """Trainiert die Modelle fuer den Cutoff ``c`` und wertet Prognose und Bestand aus."""
    f = st.cfg["forecast"]
    long, feats, mat = prep.long, prep.feats, prep.mat
    n, h, L = prep.n_series, st.horizon, st.lead_time
    lag_col = f"lag_{h}"
    cutoff, end = prep.dates[c], prep.dates[c + h - 1]
    test = long[(long["date"] >= cutoff) & (long["date"] <= end)]
    in_train = (long["date"] >= cutoff - np.timedelta64(f["train_days"], "D")) & \
               (long["date"] < cutoff)
    train_d = long[in_train].dropna(subset=[lag_col])
    train_w = long[in_train & window_train_mask(long["date"], cutoff, L)].dropna(
        subset=[lag_col, "target_window"])
    train_mat = mat[:, :c]
    y = mat[:, c:c + h].astype(float)

    # Diagnose: Artikel ohne Verkauf im Rueckblick (Zielniveau 0 bei den Baselines)
    zero_share, zero_demand = {}, {}
    for k in st.lookbacks:
        zero = train_mat[:, -k:].sum(axis=1) == 0
        zero_share[k] = zero.mean()
        zero_demand[k] = y[zero].sum()

    # Modell 1: taegliche Quantile (nur fuer die Genauigkeitsmessung)
    qs = st.daily_quantiles
    md = fit_quantile_models(train_d[feats], train_d["sales"], qs, f["backend"], f["model_params"])
    imp_daily = feature_importance_all(md, feats, f["backend"], train_d, train_d["sales"])
    P = {q: v.reshape(n, h) for q, v in predict_quantiles(md, test[feats]).items()}
    ma = moving_average(train_mat, h, 28)

    # Modell 2: Bedarfssumme ueber Lieferzeit + 1 Tage
    qg = st.quantile_grid
    mw = fit_quantile_models(train_w[feats], train_w["target_window"], qg,
                             f["backend"], f["model_params"])
    imp_window = feature_importance_all(mw, feats, f["backend"], train_w, train_w["target_window"])
    PW = {q: v.reshape(n, h) for q, v in predict_quantiles(mw, test[feats]).items()}
    yw = test["target_window"].to_numpy().reshape(n, h)
    valid = ~np.isnan(yw)                       # am Datenende ist das Fenster unvollstaendig
    ywz = np.where(valid, yw, 0.0)

    targets = policy_targets(st, train_mat, ma, PW)
    pinball = {}
    for name, q, tgt, is_q in targets:
        if is_q:
            # float64, damit die Summe ueber die Splits nicht in float32 gebildet wird
            loss = (pinball_elements(ywz, tgt, q) * valid).sum(axis=1)
            pinball.setdefault(name, {})[q] = loss.astype(np.float64)

    lost = np.zeros((len(targets), n))
    held = np.zeros((len(targets), n))
    for k, (_, _, tgt, _) in enumerate(targets):
        for i in range(n):
            r = simulate_order_up_to(y[i], tgt[i], L)
            lost[k, i], held[k, i] = r["lost_units"], r["inventory_units_days"]

    result = SplitResult(
        err_ma=np.abs(y - ma).sum(axis=1), err_model=np.abs(y - P[0.5]).sum(axis=1),
        actual=np.abs(y).sum(axis=1),
        daily_hits={q: int((y <= P[q]).sum()) for q in qs},
        daily_pinball={q: float(pinball_elements(y, P[q], q).sum()) for q in qs},
        daily_total=y.size, pinball=pinball,
        calib_hits={q: int(((yw <= PW[q]) & valid).sum()) for q in qg},
        calib_total=int(valid.sum()),
        points=[(name, param) for name, param, _, _ in targets],
        lost=lost, held=held, demand=y.sum(axis=1), imp_daily=imp_daily, imp_window=imp_window,
        zero_share=zero_share, zero_demand=zero_demand, total_demand=y.sum(),
    )
    # Speicher freigeben, bevor der naechste Split neue Kopien anlegt
    del md, mw, train_d, train_w, test, targets, PW, P
    gc.collect()
    return result


def pool(splits: list) -> Pooled:
    """Summiert die Kennzahlen ueber die Splits (je Artikel bzw. je Kurvenpunkt und Artikel)."""
    first = splits[0]

    def total(attr):
        out = getattr(first, attr)
        for s in splits[1:]:
            out = out + getattr(s, attr)
        return out

    def total_dict(attr):
        out = dict(getattr(first, attr))
        for s in splits[1:]:
            for key, value in getattr(s, attr).items():
                out[key] = out[key] + value
        return out

    pinball = {name: dict(by_q) for name, by_q in first.pinball.items()}
    for s in splits[1:]:
        for name, by_q in s.pinball.items():
            for q, value in by_q.items():
                pinball[name][q] = pinball[name][q] + value

    zero_share = {k: 0.0 for k in first.zero_share}
    for s in splits:
        for k, value in s.zero_share.items():
            zero_share[k] += value / len(splits)

    return Pooled(
        err_ma=total("err_ma"), err_model=total("err_model"), actual=total("actual"),
        daily_hits=total_dict("daily_hits"), daily_pinball=total_dict("daily_pinball"),
        daily_total=total("daily_total"), pinball=pinball,
        calib_hits=total_dict("calib_hits"), calib_total=total("calib_total"),
        policies=np.array([p for p, _ in first.points]),
        params=np.array([a for _, a in first.points]),
        lost=np.sum([s.lost for s in splits], axis=0),
        held=np.sum([s.held for s in splits], axis=0),
        demand=np.sum([s.demand for s in splits], axis=0),
        zero_share=zero_share, zero_demand=total_dict("zero_demand"),
        total_demand=total("total_demand"), splits=list(splits),
    )
