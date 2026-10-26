import numpy as np
import pandas as pd

from dfi.features import build_features


def _toy(n=80):
    dates = pd.date_range("2020-01-01", periods=n)
    df = pd.DataFrame({
        "id": "A", "item_id": "A", "store_id": "S", "dept_id": "D",
        "date": dates, "sales": np.arange(n, dtype="float32"),
        "sell_price": 1.0, "event_name_1": None, "snap": 0,
        "wday": dates.weekday + 1, "month": dates.month,
    })
    return df


def test_lag_features_use_only_past():
    df, _ = build_features(_toy(), horizon=28)
    row = df.iloc[60]
    assert row["lag_28"] == df.iloc[60 - 28]["sales"]
    # Rolling-Feature darf den aktuellen Tag nicht enthalten
    assert row["rmean_28_7"] <= df.iloc[60 - 28]["sales"]


def test_window_target_sums_next_days_and_is_nan_at_end():
    from dfi.features import add_window_target
    df = add_window_target(_toy(10), lead_time=2)
    assert df.iloc[0]["target_window"] == 0 + 1 + 2
    assert df.iloc[5]["target_window"] == 5 + 6 + 7
    assert np.isnan(df.iloc[9]["target_window"])


def test_window_train_mask_prevents_leakage_into_test():
    from dfi.features import window_train_mask
    dates = pd.Series(pd.date_range("2020-01-01", periods=30))
    cutoff = pd.Timestamp("2020-01-20")
    lead_time = 3
    mask = window_train_mask(dates, cutoff.to_datetime64(), lead_time)
    last_start = dates[mask].max()
    assert last_start + pd.Timedelta(days=lead_time) < cutoff


def test_long_window_features_match_manual_computation_and_use_only_past():
    from dfi.features import build_features
    rng = np.random.default_rng(1)
    n, h, w = 400, 28, 91
    dates = pd.date_range("2020-01-01", periods=n)
    sales = rng.poisson(0.6, n).astype("float32")
    df = pd.DataFrame({"id": "A", "item_id": "A", "store_id": "S", "dept_id": "D", "date": dates,
                       "sales": sales, "sell_price": 1.0, "event_name_1": None, "snap": 0,
                       "wday": dates.weekday + 1, "month": dates.month})
    out, feats = build_features(df, horizon=h, long_windows=(w, 182))
    t_ = 300
    window = sales[t_ - h - w + 1: t_ - h + 1]          # endet h Tage vor t_
    row = out.iloc[t_]
    assert abs(row[f"rmean_{h}_{w}"] - window.mean()) < 1e-5
    assert abs(row[f"rzero_{h}_{w}"] - (window == 0).mean()) < 1e-5
    assert abs(row[f"rstd_{h}_{w}"] - window.std(ddof=1)) < 1e-4
    assert {f"rmean_{h}_182", f"rzero_{h}_182"} <= set(feats)


def test_required_history_covers_longest_window():
    from dfi.features import required_history
    assert required_history(28, ()) == 56
    assert required_history(28, (91, 364)) == 392
