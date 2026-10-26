import numpy as np
import pandas as pd

from dfi import plots


def test_all_plots_write_files(tmp_path):
    curve = pd.DataFrame({
        "policy": ["baseline_mean_z"] * 3 + ["quantile_model_window"] * 3,
        "param": [0, 1, 2] * 2, "fill_rate": [0.8, 0.9, 0.95, 0.85, 0.93, 0.97],
        "avg_inventory": [2.0, 4.0, 7.0, 2.2, 4.5, 8.0]})
    bands = {"baseline_mean_z": (np.array([0.8, 0.9, 0.95]), np.array([1.8, 3.6, 6.5]), np.array([2.2, 4.4, 7.5]))}
    plots.plot_tradeoff(curve, tmp_path / "a.png", bands=bands)
    plots.plot_inventory_at_sl({0.9: {"baseline_mean_z": (4.0, 3.5, 4.5),
                                      "quantile_model_window": (3.8, 3.4, 4.2)},
                                0.98: {"quantile_model_window": (9.0, 8.0, 10.0)}}, tmp_path / "b.png")
    plots.plot_pinball([("quantile_model_window", "Modell", [(0.5, 0.1, 0.05, 0.15), (0.9, 0.2, 0.1, 0.3)])], tmp_path / "c.png")
    plots.plot_costs({5: {"quantile_model_window": (-0.02, -0.04, 0.0),
                          "baseline_mean_z": (-0.05, -0.08, -0.02)}}, tmp_path / "d.png")
    plots.plot_stability(pd.DataFrame({"split": [1, 2], "baseline_mean_z": [5.0, 5.2],
                                       "quantile_model_window": [4.8, np.nan]}), tmp_path / "e.png")
    plots.plot_wape({"gleitender Durchschnitt": (0.75, 0.74, 0.76),
                     "Quantilmodell (q50)": (0.70, 0.69, 0.71)}, tmp_path / "f.png")
    rng = np.random.default_rng(0)
    plots.plot_bootstrap_histograms([
        ("Test", "x", [("A", "quantile_model_window", rng.normal(0.05, 0.02, 200), 0.05),
                       ("B", "baseline_mean_z", rng.normal(-0.02, 0.02, 200), -0.02)])],
        tmp_path / "g.png")
    feats = np.array(["lag_28", "price_rel", "wday", "item_code"])
    data = {q: (feats, np.array([0.5, 0.2, 0.2, 0.1]) + 0.01 * i, np.array([0.45, 0.15, 0.15, 0.05]),
                np.array([0.55, 0.25, 0.25, 0.15])) for i, q in enumerate([0.5, 0.9])}
    plots.plot_feature_importance(data, tmp_path / "h.png")
    assert plots.feature_label("rmean_28_7").startswith("Ø")
    for name in "abcdefgh":
        assert (tmp_path / f"{name}.png").stat().st_size > 1000


def test_set_policies_assigns_unique_styles_and_labels():
    plots.set_policies(["quantile_model_window", "quantile_model_window_floor", "baseline_mean_z",
                        "baseline_empirical_q_365_floor"])
    colors = {s["color"] for s in plots.POLICY_STYLE.values()}
    assert len(colors) == 4
    assert plots.policy_label("baseline_empirical_q_365_floor") == "Empirisches Quantil (365 T.) + Mindestbestand"
    assert plots.policy_label("quantile_model_window") == "Quantilmodell"
