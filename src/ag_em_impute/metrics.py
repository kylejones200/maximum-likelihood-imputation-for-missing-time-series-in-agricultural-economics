from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ag_em_impute.baselines import BASELINE_METHODS
from ag_em_impute.em import EMResult, expectation_maximization


def rmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    err = pred[mask] - true[mask]
    return float(np.sqrt(np.mean(err**2)))


def mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.abs(pred[mask] - true[mask])))


def evaluate_missing(
    predictions: dict[str, np.ndarray],
    true: np.ndarray,
    missing_mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    """RMSE and MAE on held-out (missing) indices only."""
    if not missing_mask.any():
        return {}
    results: dict[str, dict[str, float]] = {}
    for name, pred in predictions.items():
        results[name] = {
            "rmse": rmse(pred, true, missing_mask),
            "mae": mae(pred, true, missing_mask),
        }
    return results


def run_all_imputations(
    yields: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[EMResult, dict[str, np.ndarray]]:
    em_result = expectation_maximization(yields, max_iter=max_iter, tol=tol)
    baselines = {name: fn(yields) for name, fn in BASELINE_METHODS.items()}
    return em_result, baselines


def build_metrics_report(
    yields: np.ndarray,
    true: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, Any]:
    missing_mask = np.isnan(yields)
    em_result, baselines = run_all_imputations(yields, max_iter=max_iter, tol=tol)

    predictions = {"em": em_result.imputed, **baselines}
    scores = evaluate_missing(predictions, true, missing_mask)

    return {
        "n_missing": int(missing_mask.sum()),
        "n_observed": int((~missing_mask).sum()),
        "em": {
            "mu": em_result.mu,
            "sigma": em_result.sigma,
            "n_iter": em_result.n_iter,
            "converged": em_result.converged,
            **scores.get("em", {}),
        },
        "baselines": {name: scores[name] for name in baselines},
        "best_rmse": min(scores.items(), key=lambda kv: kv[1]["rmse"])[0] if scores else None,
    }


def save_metrics(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
