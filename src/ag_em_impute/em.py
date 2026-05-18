from dataclasses import dataclass

import numpy as np

from ag_em_impute.validate import validate_yield_array


@dataclass(frozen=True)
class EMResult:
    """Univariate Gaussian EM fit with imputed series."""

    imputed: np.ndarray
    mu: float
    sigma: float
    n_iter: int
    converged: bool

    @property
    def variance(self) -> float:
        return self.sigma**2


def expectation_maximization(data: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> EMResult:
    """
    Fill missing values via EM under i.i.d. N(mu, sigma^2).
    E-step: missing <- mu. M-step: mu, sigma^2 from the completed sample.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol <= 0:
        raise ValueError("tol must be positive")

    values = validate_yield_array(data)
    missing = np.isnan(values)
    if not missing.any():
        mu = float(np.mean(values))
        sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        return EMResult(
            imputed=values.copy(),
            mu=mu,
            sigma=sigma,
            n_iter=0,
            converged=True,
        )

    filled = values.copy()
    filled[missing] = float(np.nanmean(values))
    mu_prev = np.nan
    sigma_prev = np.nan
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        mu = float(np.mean(filled))
        if filled.size > 1:
            sigma = float(np.std(filled, ddof=1))
        else:
            sigma = 0.0

        filled[missing] = mu
        if n_iter > 1 and abs(mu - mu_prev) < tol and abs(sigma - sigma_prev) < tol:
            return EMResult(
                imputed=filled,
                mu=mu,
                sigma=sigma,
                n_iter=n_iter,
                converged=True,
            )
        mu_prev, sigma_prev = mu, sigma

    return EMResult(
        imputed=filled,
        mu=mu,
        sigma=sigma,
        n_iter=n_iter,
        converged=False,
    )
