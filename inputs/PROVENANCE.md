# Input data provenance

## `wheat_yields.csv`

Synthetic annual wheat yields for the Medium article demo.

| Setting | Value |
|---------|--------|
| Years | 2000–2019 (20 observations) |
| Trend | `3 + 0.05 × (year − 2000)` tons/hectare |
| Noise | `N(0, 0.2²)` |
| RNG | `np.random.seed(42)` |
| Missing | 5 years chosen without replacement |

Missing years (seed 42): **2007, 2009, 2010, 2012, 2019**.

## `wheat_yields_truth.csv`

Complete yields before masking. Used only for **validation metrics** (`metrics.json`); not required at inference time.

## Regenerate

```bash
uv run python scripts/generate_inputs.py
```
