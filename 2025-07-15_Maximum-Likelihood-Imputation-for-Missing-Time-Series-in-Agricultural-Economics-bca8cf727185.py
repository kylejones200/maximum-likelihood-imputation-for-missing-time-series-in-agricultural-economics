# Description: Short example for Maximum Likelihood Imputation for Missing Time Series in Agricultural Economics.



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(42)


# Simulate 20 years of wheat yield data with a trend and noise
years = np.arange(2000, 2020)
true_yields = 3 + 0.05 * (years - 2000) + np.random.normal(0, 0.2, len(years))
# Introduce missing values at 5 random locations
yields_with_missing = true_yields.copy()
missing_idx = np.random.choice(len(years), size=5, replace=False)
yields_with_missing[missing_idx] = np.nan
# Store in DataFrame

data = pd.DataFrame({
    'Year': pd.to_datetime(years, format='%Y'),
    'Yield': yields_with_missing
})


def expectation_maximization(data, max_iter=100, tol=1e-6):
    # Initialize with mean imputation
    filled = data.copy()
    filled[np.isnan(data)] = np.nanmean(data)
    for _ in range(max_iter):
        mu = np.mean(filled)
        sigma = np.std(filled)
        old = filled.copy()
        # E-step: replace missing with current expected mean
        filled[np.isnan(data)] = mu
        # M-step: new mean and std from filled data
        mu_new = np.mean(filled)
        if np.abs(mu_new - mu) < tol:
            break
    return filled

# Estimate missing values
estimated_yields = expectation_maximization(yields_with_missing)
# Add to DataFrame
data['Estimated_Yield'] = estimated_yields

# Plot results
fig, ax = plt.subplots()
ax.plot(data['Year'], data['Yield'], 'bo-', label='Observed (with missing)')
ax.plot(data['Year'], data['Estimated_Yield'], 'ro--', label='EM Estimate')
ax.set_title("Crop Yield Estimation Using Expectation-Maximization (tons per hectare)", fontsize=12)
ax.legend()

# Tufte-style formatting
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('outward', 5))
ax.spines['bottom'].set_position(('outward', 5))
ax.tick_params(axis='both', direction='out')

plt.tight_layout()
plt.savefig("crop_yield_estimation.png", dpi=300)
plt.show()
