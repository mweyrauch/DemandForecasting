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
    plots.plot_pinball([(0.5, 0.1, 0.05, 0.15), (0.9, 0.2, 0.1, 0.3)], tmp_path / "c.png")
    plots.plot_costs({5: (-0.02, -0.04, 0.0), 20: (-0.05, -0.08, -0.02)}, tmp_path / "d.png")
    feats = np.array(["lag_28", "price_rel", "wday", "item_code"])
    data = {q: (feats, np.array([0.5, 0.2, 0.2, 0.1]) + 0.01 * i, np.array([0.45, 0.15, 0.15, 0.05]),
                np.array([0.55, 0.25, 0.25, 0.15])) for i, q in enumerate([0.5, 0.9])}
    plots.plot_feature_importance(data, tmp_path / "e.png")
    assert plots.feature_label("rmean_28_7").startswith("Ø")
    for name in "abcde":
        assert (tmp_path / f"{name}.png").stat().st_size > 1000


def test_set_policies_assigns_unique_styles_and_labels():
    plots.set_policies(["quantile_model_window", "quantile_model_window_floor", "baseline_mean_z",
                        "baseline_empirical_q_365_floor"])
    colors = {s["color"] for s in plots.POLICY_STYLE.values()}
    assert len(colors) == 4
    assert plots.policy_label("baseline_empirical_q_365_floor") == "Empirisches Quantil (365 T.) + Mindestbestand"
    assert plots.policy_label("quantile_model_window") == "Quantilmodell"


def test_tradeoff_fill_max_none_draws_all_points_and_default_clips():
    curve = pd.DataFrame({
        "policy": ["quantile_model_window"] * 4,
        "param": [0, 1, 2, 3], "fill_rate": [0.90, 0.97, 0.99, 0.999],
        "avg_inventory": [2.0, 4.0, 6.0, 20.0]})
    captured = []
    original = plots._save
    plots._save = lambda fig, path: captured.append(fig)
    try:
        plots.plot_tradeoff(curve, "unused.png", fill_max=None)
        plots.plot_tradeoff(curve, "unused.png")
    finally:
        plots._save = original
    n_all = len(captured[0].axes[0].get_lines()[0].get_xdata())
    n_clipped = len(captured[1].axes[0].get_lines()[0].get_xdata())
    assert n_all == 4 and n_clipped == 3                      # Standard schneidet den Saettigungspunkt ab
    for fig in captured:
        plots.plt.close(fig)


def test_band_ends_where_the_drawn_line_ends():
    curve = pd.DataFrame({
        "policy": ["quantile_model_window"] * 4,
        "param": [0, 1, 2, 3], "fill_rate": [0.90, 0.97, 0.99, 0.999],
        "avg_inventory": [2.0, 4.0, 6.0, 20.0]})
    grid = np.linspace(0.90, 0.999, 12)
    bands = {"quantile_model_window": (grid, np.linspace(2.0, 19.0, 12), np.linspace(2.5, 21.0, 12))}
    captured = []
    original = plots._save
    plots._save = lambda fig, path: captured.append(fig)
    try:
        plots.plot_tradeoff(curve, "unused.png", bands=bands)                  # Standard: bis 99,5 %
        plots.plot_tradeoff(curve, "unused.png", bands=bands, fill_max=None)   # alle Punkte
    finally:
        plots._save = original

    def band_top(fig):
        ax = fig.axes[0]
        return max(path.vertices[:, 1].max() for coll in ax.collections for path in coll.get_paths())

    def line_top(fig):
        return max(np.max(ln.get_ydata()) for ln in fig.axes[0].get_lines())

    assert abs(band_top(captured[0]) - 0.99) < 1e-6 and abs(line_top(captured[0]) - 0.99) < 1e-6
    assert abs(band_top(captured[1]) - 0.999) < 1e-6 and abs(line_top(captured[1]) - 0.999) < 1e-6
    for fig in captured:
        plots.plt.close(fig)
