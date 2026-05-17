#!/usr/bin/env python3
"""Regenerate inputs (seed=42, matches the Medium article)."""

from ag_em_impute.paths import INPUTS_DIR
from ag_em_impute.simulate import generate_wheat_yields


def main() -> None:
    yields_path = INPUTS_DIR / "wheat_yields.csv"
    truth_path = INPUTS_DIR / "wheat_yields_truth.csv"
    generate_wheat_yields(yields_path, truth_path=truth_path)
    print(f"Wrote {yields_path}")
    print(f"Wrote {truth_path}")


if __name__ == "__main__":
    main()
