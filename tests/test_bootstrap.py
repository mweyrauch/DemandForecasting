import numpy as np

from dfi.bootstrap import bootstrap_counts, curve_arrays, interval, inventory_at_levels, min_cost


def test_bootstrap_counts_sum_to_n_and_are_reproducible():
    c1 = bootstrap_counts(50, 20, seed=1)
    c2 = bootstrap_counts(50, 20, seed=1)
    assert (c1.sum(axis=1) == 50).all()
    assert (c1 == c2).all()


def test_interval_ignores_nan():
    lo, hi = interval([1, 2, 3, 4, np.nan])
    assert 1 <= lo <= hi <= 4


def test_curve_arrays_with_unit_weights_matches_direct_computation():
    lost = np.array([[1.0, 3.0], [0.0, 0.0]])        # 2 Politikpunkte, 2 Artikel
    held = np.array([[10.0, 10.0], [30.0, 30.0]])
    demand = np.array([50.0, 50.0])
    fill, inv = curve_arrays(np.ones((1, 2)), lost, held, demand, days=10)
    assert abs(fill[0, 0] - 0.96) < 1e-9 and fill[0, 1] == 1.0
    assert abs(inv[0, 0] - 1.0) < 1e-9 and abs(inv[0, 1] - 3.0) < 1e-9


def test_inventory_at_levels_and_min_cost_per_policy():
    policies = ["a", "a", "b", "b"]
    fill = np.array([[0.8, 1.0, 0.8, 1.0]])
    inv = np.array([[1.0, 5.0, 2.0, 6.0]])
    res = inventory_at_levels(fill, inv, policies, 0.9)
    assert res["a"][0] == 3.0 and res["b"][0] == 4.0
    lost = np.array([[10.0], [0.0], [10.0], [0.0]])
    held = np.array([[10.0], [50.0], [20.0], [60.0]])
    cost = min_cost(np.ones((1, 1)), lost, held, policies, ratio=1)
    assert cost["a"][0] == 20.0 and cost["b"][0] == 30.0


def test_curve_bands_are_ordered_and_bracket_point_estimate():
    from dfi.bootstrap import curve_bands
    rng = np.random.default_rng(0)
    B = 60
    base_fill = np.array([0.8, 0.9, 0.97])
    base_inv = np.array([2.0, 4.0, 8.0])
    fill = base_fill + rng.normal(0, 0.003, (B, 3))
    inv = base_inv + rng.normal(0, 0.2, (B, 3))
    bands = curve_bands(fill, inv, base_fill, ["a", "a", "a"], n_grid=10)
    grid, lo, hi = bands["a"]
    assert len(grid) > 0 and (lo <= hi).all()
    mid = np.interp(grid, base_fill, base_inv)
    assert ((lo - 0.5 <= mid) & (mid <= hi + 0.5)).all()
