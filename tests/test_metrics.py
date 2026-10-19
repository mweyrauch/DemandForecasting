import numpy as np

from dfi.evaluate import calibration, pinball_loss, rolling_origin_cutoffs, wape


def test_wape_perfect_and_simple():
    assert wape([1, 2, 3], [1, 2, 3]) == 0
    assert abs(wape([2, 2], [1, 3]) - 0.5) < 1e-9


def test_pinball_median_equals_half_mae():
    y, p = np.array([1.0, 3.0]), np.array([2.0, 2.0])
    assert abs(pinball_loss(y, p, 0.5) - 0.5) < 1e-9


def test_calibration_fraction():
    assert calibration([1, 2, 3, 4], [2, 2, 2, 2]) == 0.5


def test_rolling_cutoffs_are_ordered_and_in_range():
    cuts = rolling_origin_cutoffs(100, 28, 3)
    assert cuts == sorted(cuts)
    assert max(cuts) + 28 <= 100
