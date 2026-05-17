import json

import numpy as np
import pytest

from ag_em_impute.baselines import (
    forward_fill,
    linear_interpolation,
    mean_imputation,
)
from ag_em_impute.metrics import build_metrics_report, evaluate_missing
from ag_em_impute.paths import INPUTS_DIR
from ag_em_impute.runner import run


def test_baselines_fill_all_entries():
    values = np.array([1.0, np.nan, 3.0, np.nan])
    for fn in (mean_imputation, forward_fill, linear_interpolation):
        filled = fn(values)
        assert not np.isnan(filled).any()


def test_evaluate_missing_only_scores_masked_cells():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    missing = np.array([False, True, False, True])
    preds = {"a": np.array([1.0, 2.5, 3.0, 3.5])}
    scores = evaluate_missing(preds, true, missing)
    assert scores["a"]["rmse"] == pytest.approx(0.5)
    assert scores["a"]["mae"] == pytest.approx(0.5)


def test_build_metrics_report_orders_methods():
    yields = np.array([1.0, np.nan, 3.0, np.nan])
    true = np.array([1.0, 2.0, 3.0, 4.0])
    report = build_metrics_report(yields, true)
    assert report["n_missing"] == 2
    assert "em" in report
    assert "mean" in report["baselines"]
    assert report["best_rmse"] in report["baselines"] or report["best_rmse"] == "em"


def test_wheat_validation_metrics(tmp_path):
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
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.dump(config))
    run(config_path, validate=True)

    report = json.loads((tmp_path / "metrics.json").read_text())
    assert report["n_missing"] == 5
    assert report["em"]["converged"] is True
    # EM ties mean (global level); trend-aware baselines win on this series.
    assert report["em"]["rmse"] == pytest.approx(
        report["baselines"]["mean"]["rmse"], rel=1e-9
    )
    assert report["best_rmse"] in {"forward_fill", "linear_interp"}
    assert report["baselines"]["forward_fill"]["rmse"] < report["em"]["rmse"]
