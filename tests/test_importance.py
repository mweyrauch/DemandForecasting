import types

import numpy as np
import pandas as pd

from dfi.models import feature_importance, fit_quantile_models


def test_lightgbm_branch_normalizes_gain():
    fake = types.SimpleNamespace(
        booster_=types.SimpleNamespace(feature_importance=lambda importance_type: [3.0, 1.0, 0.0]))
    imp = feature_importance(fake, ["a", "b", "c"], "lightgbm")
    assert abs(imp.sum() - 1.0) < 1e-9 and imp[0] == 0.75


def test_sklearn_permutation_finds_informative_feature():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"signal": rng.normal(size=600), "noise": rng.normal(size=600)})
    y = 3 * X["signal"] + rng.normal(scale=0.1, size=600)
    models = fit_quantile_models(X, y, [0.5], backend="sklearn", params={"n_estimators": 50})
    imp = feature_importance(models[0.5], ["signal", "noise"], "sklearn", X, y, q=0.5)
    assert imp[0] > imp[1] and abs(imp.sum() - 1.0) < 1e-9
