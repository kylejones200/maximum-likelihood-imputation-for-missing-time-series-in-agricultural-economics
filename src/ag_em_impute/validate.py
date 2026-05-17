import numpy as np


def validate_yield_array(data: np.ndarray) -> np.ndarray:
    """Ensure a 1-D float array suitable for imputation."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("Cannot impute an empty series")
    observed = arr[~np.isnan(arr)]
    if observed.size == 0:
        raise ValueError("All values are missing; cannot initialize imputation")
    if not np.all(np.isfinite(observed)):
        raise ValueError("Observed values must be finite (no inf)")
    return arr
