from __future__ import annotations

import logging
from pathlib import Path

import yaml

from ag_em_impute.em import expectation_maximization
from ag_em_impute.io import load_truth, load_yields, save_results
from ag_em_impute.metrics import build_metrics_report, save_metrics
from ag_em_impute.paths import INPUTS_DIR, ROOT
from ag_em_impute.plot import plot_yield_imputation
from ag_em_impute.simulate import generate_wheat_yields

logger = logging.getLogger(__name__)


def load_config(config_path: Path | None = None) -> dict:
    path = config_path or ROOT / "config.yaml"
    with path.open() as handle:
        return yaml.safe_load(handle)


def _resolve_path(config_value: str, override: Path | None) -> Path:
    if override is not None:
        return override
    path = Path(config_value)
    return path if path.is_absolute() else ROOT / path


def run(
    config_path: Path | None = None,
    *,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    show_plots: bool | None = None,
    validate: bool | None = None,
) -> Path:
    config = load_config(config_path)
    yields_path = _resolve_path(config["input"]["yields_csv"], input_path)
    out_root = _resolve_path(config["output"]["dir"], output_dir)
    figures_dir = out_root / config["output"]["figures_dir"]

    do_validate = (
        validate if validate is not None else config.get("validation", {}).get("enabled", False)
    )
    truth_path_cfg = config.get("validation", {}).get("truth_csv")
    truth_path = _resolve_path(truth_path_cfg, None) if truth_path_cfg else None

    data = load_yields(yields_path)
    yields = data["Yield"].to_numpy(dtype=float)
    em_result = expectation_maximization(
        yields,
        max_iter=config["model"]["max_iter"],
        tol=config["model"]["tol"],
    )

    data["Estimated_Yield"] = em_result.imputed
    data["Was_Missing"] = data["Yield"].isna()

    save_results(data, out_root / config["output"]["results_csv"])
    logger.info(
        "EM: mu=%.4f sigma=%.4f iterations=%d converged=%s",
        em_result.mu,
        em_result.sigma,
        em_result.n_iter,
        em_result.converged,
    )

    true_series = None
    if do_validate:
        if truth_path is None or not truth_path.exists():
            raise FileNotFoundError(f"Validation enabled but truth file not found: {truth_path}")
        true_series = load_truth(truth_path)
        true_values = true_series.reindex(data["Year"]).to_numpy(dtype=float)
        report = build_metrics_report(
            yields,
            true_values,
            max_iter=config["model"]["max_iter"],
            tol=config["model"]["tol"],
        )
        metrics_path = out_root / config["output"].get("metrics_json", "metrics.json")
        save_metrics(report, metrics_path)
        logger.info("Validation metrics written to %s", metrics_path)
        logger.info("Best RMSE on missing cells: %s", report.get("best_rmse"))

    plot_yield_imputation(
        data,
        figures_dir / config["output"]["figure_name"],
        true_yields=true_series.reindex(data["Year"]) if true_series is not None else None,
        dpi=config["output"]["figure_dpi"],
        show=show_plots if show_plots is not None else config["run"]["show_plots"],
    )
    logger.info("Outputs written to %s", out_root)
    return out_root


def ensure_inputs(inputs_dir: Path | None = None) -> Path:
    """Create synthetic wheat yields (and truth) if missing."""
    base = inputs_dir or INPUTS_DIR
    target = base / "wheat_yields.csv"
    truth = base / "wheat_yields_truth.csv"
    if not target.exists() or not truth.exists():
        generate_wheat_yields(target, truth_path=truth)
    return target
