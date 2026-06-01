# Inputs

| File | Description |
|------|-------------|
| `wheat_yields.csv` | Observed yields with five missing values |
| `wheat_yields_truth.csv` | Full series for validation scoring only |

See [`PROVENANCE.md`](PROVENANCE.md) for simulation details.

```bash
uv run python scripts/generate_inputs.py
```
