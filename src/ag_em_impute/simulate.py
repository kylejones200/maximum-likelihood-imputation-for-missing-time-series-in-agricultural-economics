from pathlib import Path

import numpy as np
import pandas as pd


def generate_wheat_yields(
    output_path: Path,
    *,
    truth_path: Path | None = None,
    seed: int = 42,
    start_year: int = 2000,
    n_years: int = 20,
    n_missing: int = 5,
) -> pd.DataFrame:
    """Simulate wheat yields with random missing values (article example)."""
    np.random.seed(seed)
    years = np.arange(start_year, start_year + n_years)
    true_yields = 3 + 0.05 * (years - start_year) + np.random.normal(0, 0.2, len(years))
    yields_with_missing = true_yields.copy()
    missing_idx = np.random.choice(len(years), size=n_missing, replace=False)
    yields_with_missing[missing_idx] = np.nan

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Year": years, "Yield": yields_with_missing}).to_csv(output_path, index=False)

    if truth_path is not None:
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Year": years, "True_Yield": true_yields}).to_csv(truth_path, index=False)

    return pd.DataFrame({"Year": years, "Yield": yields_with_missing})
