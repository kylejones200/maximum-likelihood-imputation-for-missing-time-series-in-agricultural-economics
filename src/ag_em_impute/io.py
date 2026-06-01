from pathlib import Path

import pandas as pd


def load_yields(path: Path) -> pd.DataFrame:
    """Load year and yield columns; parse years as datetimes."""
    df = pd.read_csv(path)
    if "Year" not in df.columns or "Yield" not in df.columns:
        raise ValueError(f"{path} must contain Year and Yield columns")
    df["Year"] = pd.to_datetime(df["Year"], format="%Y")
    return df


def load_truth(path: Path) -> pd.Series:
    """Load true yields indexed by year (for validation only)."""
    df = pd.read_csv(path)
    if "Year" not in df.columns or "True_Yield" not in df.columns:
        raise ValueError(f"{path} must contain Year and True_Yield columns")
    df["Year"] = pd.to_datetime(df["Year"], format="%Y")
    return df.set_index("Year")["True_Yield"]


def save_results(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if pd.api.types.is_datetime64_any_dtype(out["Year"]):
        out["Year"] = out["Year"].dt.year
    out.to_csv(path, index=False)
