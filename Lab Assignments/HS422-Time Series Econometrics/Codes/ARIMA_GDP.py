import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import ccf

# Load GDP and NIFTY data
gdp_data = pd.read_csv("../Data/gdp_data.csv", index_col="Year")
nifty_data = pd.read_csv("../Data/nifty_data.csv", index_col="Year")

# Flatten the data to create single time-series for GDP and NIFTY
gdp_values = gdp_data.values.flatten()
nifty_values = nifty_data.values.flatten()

# Ensure alignment of both series
assert len(gdp_values) == len(nifty_values), "GDP and NIFTY data must have the same length."

# Fit ARIMA model on GDP data
arima_model = ARIMA(gdp_values, order=(1, 1, 1))  # ARIMA(1,1,1) model
arima_result = arima_model.fit()

# Print summary of the ARIMA model
print(arima_result.summary())

# Extract fitted values and residuals
fitted_values = arima_result.fittedvalues
residuals = arima_result.resid

# Cross-correlation between ARIMA residuals and NIFTY returns
cross_corr = ccf(residuals, nifty_values, adjusted=False)

import numpy as np

# Generate year labels for every 4th data point (quarters -> years)
years = list(range(2016, 2024))
x_ticks = np.arange(0, len(gdp_values), 4)  # Tick positions corresponding to yearly intervals

plt.figure(figsize=(12, 6))

# Plot GDP, fitted values, and residuals
plt.subplot(2, 1, 1)
plt.plot(gdp_values, label="Actual GDP", color="blue")
plt.plot(fitted_values, label="Fitted GDP (ARIMA)", color="red")
plt.title("GDP and ARIMA Fitted Values")
plt.xticks(ticks=x_ticks, labels=years, rotation=45)  # Set year labels for every 4th data point
plt.xlabel("Year")
plt.ylabel("GDP Return (%)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(residuals, label="GDP Residuals", color="purple")
plt.title("Residuals of ARIMA Model (GDP)")
plt.xticks(ticks=x_ticks, labels=years, rotation=45)  # Set year labels for every 4th data point
plt.xlabel("Year")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

# Cross-correlation between residuals and NIFTY returns
plt.figure(figsize=(8, 4))
plt.stem(cross_corr, use_line_collection=True)
plt.title("Cross-Correlation between GDP Residuals and NIFTY Returns")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.show()