import numpy as np

from dfi.inventory import simulate_order_up_to, window_sum


def test_window_sum():
    assert list(window_sum([1, 2, 3, 4], 2)) == [3, 5, 7, 4]


def test_high_target_gives_full_service():
    demand = np.full(30, 5.0)
    res = simulate_order_up_to(demand, np.full(30, 1000.0), lead_time=2)
    assert res["fill_rate"] == 1.0


def test_zero_target_loses_everything():
    demand = np.full(30, 5.0)
    res = simulate_order_up_to(demand, np.zeros(30), lead_time=2)
    assert res["fill_rate"] == 0.0


def test_empirical_quantile_targets_shape_and_level():
    import numpy as np
    from dfi.inventory import empirical_quantile_targets
    train = np.tile(np.array([[1.0, 2.0, 3.0, 4.0]]), (1, 25))  # 100 Tage
    tgt = empirical_quantile_targets(train, window=4, q=0.5, n_days=7, emp_days=40)
    assert tgt.shape == (1, 7)
    assert tgt[0, 0] == 10.0  # jede 4-Tage-Summe ist 10


def test_inventory_at_service_level_interpolates_and_does_not_extrapolate():
    import math
    from dfi.inventory import inventory_at_service_level
    assert abs(inventory_at_service_level([0.8, 0.9, 1.0], [2, 4, 8], 0.95) - 6.0) < 1e-9
    assert math.isnan(inventory_at_service_level([0.8, 0.9], [2, 4], 0.95))


def test_best_cost_by_ratio_picks_cheapest_parameter():
    import pandas as pd
    from dfi.inventory import best_cost_by_ratio
    res = pd.DataFrame({
        "policy": ["p", "p"], "param": [1, 2], "fill_rate": [0.8, 0.99],
        "avg_inventory": [1.0, 5.0], "inventory_units_days": [100.0, 500.0],
        "lost_units": [50.0, 1.0],
    })
    low = best_cost_by_ratio(res, [1]).iloc[0]    # Fehlmengen billig -> kleines Lager
    high = best_cost_by_ratio(res, [100]).iloc[0]  # Fehlmengen teuer -> grosses Lager
    assert low["best_param"] == 1 and high["best_param"] == 2
