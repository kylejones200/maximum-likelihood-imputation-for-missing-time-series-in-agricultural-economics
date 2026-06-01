import numpy as np
import pandas as pd

from ag_em_impute.validate import validate_yield_array


def mean_imputation(data: np.ndarray) -> np.ndarray:
    """Fill missing values with the mean of observed points."""
    values = validate_yield_array(data)
    filled = values.copy()
    filled[np.isnan(values)] = float(np.nanmean(values))
    return filled


def forward_fill(data: np.ndarray) -> np.ndarray:
    """Carry the last observed value forward (pandas ffill)."""
    values = validate_yield_array(data)
    series = pd.Series(values).ffill()
    if series.isna().any():
        series = series.bfill()
    return series.to_numpy(dtype=float)


def linear_interpolation(data: np.ndarray) -> np.ndarray:
    """Linear interpolation over the index; edges use nearest observed."""
    values = validate_yield_array(data)
    series = pd.Series(values).interpolate(method="linear")
    if series.isna().any():
        series = series.ffill().bfill()
    return series.to_numpy(dtype=float)


BASELINE_METHODS = {
    "mean": mean_imputation,
    "forward_fill": forward_fill,
    "linear_interp": linear_interpolation,
}
