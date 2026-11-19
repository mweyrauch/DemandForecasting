from dataclasses import replace

import numpy as np

from dfi.config import deep_merge, load_config
from dfi.pipeline import MODEL, Settings, SplitResult, pool


def test_load_config_applies_only_given_overrides():
    cfg = load_config("configs/default.yaml", stores=["TX_1"], n_splits=None, step_days=91)
    assert cfg["data"]["stores"] == ["TX_1"]
    assert cfg["evaluation"]["step_days"] == 91
    assert cfg["evaluation"]["n_splits"] == load_config("configs/default.yaml")["evaluation"]["n_splits"]


def test_settings_derive_policies_and_window():
    cfg = load_config("configs/synthetic.yaml")
    st = Settings.from_config(cfg)
    assert st.window == st.lead_time + 1 and st.step == st.horizon
    assert st.policy_names == [MODEL, "baseline_mean_z_floor", "baseline_empirical_q_84_floor",
                               "baseline_empirical_q_365_floor"]
    assert st.policy_names[-1] == "baseline_empirical_q_365_floor"
    assert st.lookbacks == [28, 84, 365]


def _split(scale):
    return SplitResult(
        err_ma=np.array([1.0, 2.0]) * scale, err_model=np.array([0.5, 1.0]) * scale,
        actual=np.array([4.0, 6.0]) * scale, daily_hits={0.5: 3 * scale}, daily_pinball={0.5: 0.2 * scale},
        daily_total=10 * scale, pinball={MODEL: {0.9: np.array([1.0, 1.0]) * scale}},
        calib_hits={0.9: 7 * scale}, calib_total=8 * scale, points=[(MODEL, 0.9), ("b", 1.0)],
        lost=np.ones((2, 2)) * scale, held=np.ones((2, 2)) * 2 * scale, demand=np.array([4.0, 6.0]) * scale,
        imp_daily={}, imp_window={}, zero_share={28: 0.1 * scale}, zero_demand={28: 1.0 * scale},
        total_demand=10.0 * scale)


def test_pool_sums_splits_and_averages_zero_share():
    res = pool([_split(1), _split(2)])
    assert res.n_splits == 2 and list(res.policies) == [MODEL, "b"]
    assert np.allclose(res.err_ma, [3.0, 6.0]) and np.allclose(res.lost, 3.0)
    assert res.daily_hits == {0.5: 9} and res.calib_total == 24
    assert np.allclose(res.pinball[MODEL][0.9], [3.0, 3.0])
    assert abs(res.zero_share[28] - 0.15) < 1e-12        # Mittel ueber die Splits
    assert res.zero_demand[28] == 3.0 and res.total_demand == 30.0


def test_pool_does_not_modify_the_split_results():
    first = _split(1)
    before = replace(first, pinball={MODEL: {0.9: first.pinball[MODEL][0.9].copy()}})
    pool([first, _split(2)])
    assert np.array_equal(first.pinball[MODEL][0.9], before.pinball[MODEL][0.9])
    assert first.daily_hits == {0.5: 3}


def test_deep_merge_merges_sections_and_replaces_lists():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    merged = deep_merge(base, {"a": {"y": [3]}, "c": 2})
    assert merged == {"a": {"x": 1, "y": [3]}, "b": 1, "c": 2}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 1}      # Basis bleibt unveraendert


def test_synthetic_config_extends_default():
    default, synthetic = load_config("configs/default.yaml"), load_config("configs/synthetic.yaml")
    assert "extends" not in synthetic
    assert synthetic["forecast"]["backend"] == "sklearn"
    assert synthetic["forecast"]["model_params"]["num_leaves"] == default["forecast"]["model_params"]["num_leaves"]
    assert synthetic["inventory"]["baselines"] == default["inventory"]["baselines"]
