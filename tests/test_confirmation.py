import json

import pandas as pd

from dfi import plots
from dfi.evaluate import rolling_origin_cutoffs
from dfi.summary import collect_key_metrics, run_label


def test_cutoffs_with_step_cover_a_longer_period():
    cuts = rolling_origin_cutoffs(1941, 28, 4, step=91)
    assert cuts == [1640, 1731, 1822, 1913]
    assert rolling_origin_cutoffs(1941, 28, 3) == [1857, 1885, 1913]


def _write_run(path, est):
    path.mkdir(parents=True)
    rows = [
        {"metric": "inventory_rel_diff", "comparison": "quantile_model_window vs best baseline (x), SL=0.95",
         "estimate": est, "ci_low": est - 0.02, "ci_high": est + 0.02},
        {"metric": "inventory_rel_diff", "comparison": "quantile_model_window_floor vs best baseline (x), SL=0.95",
         "estimate": 9.0, "ci_low": 9.0, "ci_high": 9.0},
        {"metric": "cost_rel_diff", "comparison": "quantile_model_window vs best baseline (x), r=20",
         "estimate": est, "ci_low": est - 0.01, "ci_high": est + 0.01},
        {"metric": "pinball_improvement", "comparison": "quantile_model_window vs best empirical (x), q=0.9",
         "estimate": -est, "ci_low": -est - 0.01, "ci_high": -est + 0.01},
    ]
    pd.DataFrame(rows).to_csv(path / "summary_ci.csv", index=False)
    (path / "run_info.json").write_text(json.dumps(
        {"stores": ["CA_2"], "test_periods": [["2016-02-29", "2016-03-27"], ["2016-04-25", "2016-05-22"]]}))


def test_collect_key_metrics_picks_plain_model_rows_only(tmp_path):
    _write_run(tmp_path / "a", -0.10)
    _write_run(tmp_path / "b", -0.05)
    df = collect_key_metrics([tmp_path / "a", tmp_path / "b"])
    assert len(df) == 6                       # 3 Kennzahlen x 2 Laeufe, keine _floor-Zeilen
    assert (df["estimate"] != 9.0).all()
    assert run_label(tmp_path / "a") == "CA_2, 2016-02 bis 2016-05"


def test_forest_plot_writes_file(tmp_path):
    panels = [("Titel", "x", [("Lauf A", -0.1, -0.12, -0.08), ("Lauf B", -0.05, -0.08, -0.01)])]
    plots.plot_forest(panels, tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 1000


def test_build_command_maps_arguments():
    from dfi.summary import build_command
    spec = {"name": "x", "out_dir": "results/x", "args": {"stores": ["CA_2", "TX_1"], "step_days": 91}}
    cmd = build_command(spec, "cfg.yaml", "python", "s.py")
    assert cmd[:6] == ["python", "s.py", "--config", "cfg.yaml", "--out-dir", "results/x"]
    assert cmd[6:] == ["--stores", "CA_2", "TX_1", "--step-days", "91"]


def test_missing_runs_and_default_specs(tmp_path):
    from dfi.summary import load_run_specs, missing_runs
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "summary_ci.csv").write_text("metric\n")
    assert missing_runs(["a", "b"], root=tmp_path) == ["b"]
    specs = load_run_specs("configs/confirmation.yaml")
    assert specs[0]["out_dir"] == "results" and len(specs) >= 3
