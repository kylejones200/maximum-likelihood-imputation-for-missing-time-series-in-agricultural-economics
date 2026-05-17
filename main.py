#!/usr/bin/env python3
"""Run EM imputation on wheat yield time series."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ag_em_impute.runner import ensure_inputs, run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maximum-likelihood (EM) imputation for agricultural yields"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--input", type=Path, default=None, help="Override yields CSV")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output dir")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Score imputations vs inputs/wheat_yields_truth.csv",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation even if enabled in config",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively",
    )
    args = parser.parse_args()

    validate = None
    if args.validate:
        validate = True
    elif args.no_validate:
        validate = False

    ensure_inputs()
    output_dir = run(
        args.config,
        input_path=args.input,
        output_dir=args.output_dir,
        show_plots=args.show if args.show else None,
        validate=validate,
    )
    print(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
