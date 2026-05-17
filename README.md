# Maximum Likelihood Imputation for Missing Time Series in Agricultural Economics

Companion code for [`article.md`](article.md).

Published: 2025-07-15  
Medium: [Maximum Likelihood Imputation for Missing Time Series in Agricultural Economics](https://medium.com/@kyle-t-jones/maximum-likelihood-imputation-for-missing-time-series-in-agricultural-economics-bca8cf727185)

## Method

Univariate **Gaussian EM** for i.i.d. \(N(\mu, \sigma^2)\) observations with missing values:

1. **E-step:** impute missing cells with the current \(\mu\).
2. **M-step:** update \(\mu\) and \(\sigma\) from the completed sample.
3. Repeat until \(\mu\) and \(\sigma\) stabilize.

The demo also scores **mean imputation**, **forward fill**, and **linear interpolation** against held-out truth on the synthetic wheat series. See [`inputs/PROVENANCE.md`](inputs/PROVENANCE.md).

**Limitation:** this model ignores trend and autocorrelation; a trending series like wheat yields is better served by time-aware methods (Kalman, GP, detrended EM). The validation block makes that tradeoff visible in `metrics.json`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync --extra dev
```

## Run

```bash
uv run python main.py
# or
uv run ag-em-impute
```

With validation metrics and truth overlay on the figure (default in `config.yaml`):

```bash
uv run python main.py --validate
```

Overrides:

```bash
uv run python main.py --input inputs/wheat_yields.csv --output-dir outputs --show
```

Regenerate synthetic inputs (`np.random.seed(42)`):

```bash
uv run python scripts/generate_inputs.py
```

## Project layout

```
.
├── article.md
├── config.yaml
├── main.py
├── inputs/              # yields + truth (validation)
├── src/ag_em_impute/    # EM, baselines, metrics, I/O, plots
├── scripts/
├── outputs/             # gitignored artifacts
└── tests/
```

## Outputs

| Artifact | Path |
|----------|------|
| Imputed series | `outputs/imputed_yields.csv` |
| Validation scores | `outputs/metrics.json` |
| Figure | `outputs/figures/crop_yield_estimation.png` |

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests scripts main.py
uv run ruff format src tests scripts main.py
```

## License

MIT — see [LICENSE](LICENSE).
