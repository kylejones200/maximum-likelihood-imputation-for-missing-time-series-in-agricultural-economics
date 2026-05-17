import numpy as np
import pandas as pd
import pytest

from ag_em_impute.em import expectation_maximization
from ag_em_impute.io import load_yields
from ag_em_impute.paths import INPUTS_DIR
from ag_em_impute.runner import run
from ag_em_impute.validate import validate_yield_array

# Regression snapshot: wheat_yields.csv, seed=42, Gaussian EM (mu + sigma).
WHEAT_MU = 3.4019705749467497
WHEAT_SIGMA = 0.21797128868990173
WHEAT_N_ITER = 2
WHEAT_ESTIMATED = [
    3.099343,
    3.022347,
    3.229538,
    3.454606,
    3.153169,
    3.203173,
    3.615843,
    3.401971,
    3.306105,
    3.401971,
    3.401971,
    3.456854,
    3.401971,
    3.267344,
    3.355016,
    3.637542,
    3.597434,
    3.912849,
    3.718395,
    3.401971,
]


def test_known_answer_toy_series():
    result = expectation_maximization(np.array([1.0, np.nan, 3.0, np.nan]))
    assert result.converged
    assert result.n_iter == 2
    assert result.mu == pytest.approx(2.0)
    assert result.sigma == pytest.approx(np.sqrt(2.0 / 3.0))
    np.testing.assert_allclose(result.imputed, [1.0, 2.0, 3.0, 2.0])


def test_wheat_regression_snapshot():
    df = load_yields(INPUTS_DIR / "wheat_yields.csv")
    result = expectation_maximization(df["Yield"].to_numpy())
    assert result.converged
    assert result.n_iter == WHEAT_N_ITER
    assert result.mu == pytest.approx(WHEAT_MU)
    assert result.sigma == pytest.approx(WHEAT_SIGMA)
    np.testing.assert_allclose(np.round(result.imputed, 6), WHEAT_ESTIMATED, rtol=0, atol=0)


def test_no_missing_returns_observed():
    values = np.array([1.0, 2.0, 3.0])
    result = expectation_maximization(values)
    assert result.n_iter == 0
    assert result.converged
    np.testing.assert_array_equal(result.imputed, values)
    assert result.mu == pytest.approx(2.0)
    assert result.sigma == pytest.approx(1.0)


def test_all_missing_raises():
    with pytest.raises(ValueError, match="All values are missing"):
        expectation_maximization(np.array([np.nan, np.nan]))


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_yield_array(np.array([]))


def test_non_finite_observed_raises():
    with pytest.raises(ValueError, match="finite"):
        validate_yield_array(np.array([1.0, np.inf]))


def test_load_yields_parses_dates():
    df = load_yields(INPUTS_DIR / "wheat_yields.csv")
    assert pd.api.types.is_datetime64_any_dtype(df["Year"])
    assert df["Yield"].isna().sum() == 5


def test_run_writes_outputs_and_metrics(tmp_path):
    import yaml

    config = {
        "input": {"yields_csv": str(INPUTS_DIR / "wheat_yields.csv")},
        "output": {
            "dir": str(tmp_path),
            "figures_dir": "figures",
            "results_csv": "imputed_yields.csv",
            "metrics_json": "metrics.json",
            "figure_name": "crop_yield_estimation.png",
            "figure_dpi": 100,
        },
        "model": {"max_iter": 100, "tol": 1e-6},
        "validation": {
            "enabled": True,
            "truth_csv": str(INPUTS_DIR / "wheat_yields_truth.csv"),
        },
        "run": {"show_plots": False},
    }
    config_path = tmp_path / "test_config.yaml"
    with config_path.open("w") as handle:
        yaml.dump(config, handle)

    result_dir = run(config_path, validate=True)
    assert (result_dir / "imputed_yields.csv").exists()
    assert (result_dir / "metrics.json").exists()
    assert (result_dir / "figures" / "crop_yield_estimation.png").exists()
