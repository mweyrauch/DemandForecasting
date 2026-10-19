import numpy as np
import pandas as pd

from dfi.data import load_m5, to_long


def _write_fake(tmp_path, n_per_dept=100, n_days=40):
    rows = []
    for dept, cat in (("FOODS_1", "FOODS"), ("HOBBIES_1", "HOBBIES")):
        for i in range(n_per_dept):
            r = {"id": f"{dept}_{i}", "item_id": f"{dept}_{i}", "dept_id": dept, "cat_id": cat,
                 "store_id": "CA_1", "state_id": "CA"}
            r.update({f"d_{j + 1}": 1 for j in range(n_days)})
            rows.append(r)
    pd.DataFrame(rows).to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    dates = pd.date_range("2020-01-01", periods=n_days)
    pd.DataFrame({"date": dates, "wm_yr_wk": 1 + np.arange(n_days) // 7, "wday": 1, "month": 1,
                  "year": 2020, "d": [f"d_{j + 1}" for j in range(n_days)],
                  "event_name_1": None, "snap_CA": 0, "snap_TX": 0, "snap_WI": 0}
                 ).to_csv(tmp_path / "calendar.csv", index=False)
    pd.DataFrame({"store_id": [], "item_id": [], "wm_yr_wk": [], "sell_price": []}
                 ).to_csv(tmp_path / "sell_prices.csv", index=False)


def test_stratified_sampling_keeps_departments(tmp_path):
    _write_fake(tmp_path)
    sales, _, _ = load_m5(tmp_path, max_series=50, seed=0)
    counts = sales["dept_id"].value_counts()
    assert abs(len(sales) - 50) <= 2
    assert set(counts.index) == {"FOODS_1", "HOBBIES_1"}


def test_no_limit_returns_all(tmp_path):
    _write_fake(tmp_path)
    sales, _, _ = load_m5(tmp_path, max_series=None)
    assert len(sales) == 200


def test_last_days_limits_history(tmp_path):
    _write_fake(tmp_path)
    sales, cal, prices = load_m5(tmp_path)
    long = to_long(sales, cal, prices, last_days=10)
    assert long.groupby("id").size().eq(10).all()
