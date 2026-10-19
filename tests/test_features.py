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
