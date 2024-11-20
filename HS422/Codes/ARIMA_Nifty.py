import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the NIFTY quarterly return data (percentage returns)
nifty_data = pd.read_csv("../Data/nifty_data.csv", index_col="Year")

# Flatten the data to create a time series
nifty_returns = nifty_data.values.flatten()

# Step 1: Check if the series is stationary using Augmented Dickey-Fuller (ADF) test
adf_result = adfuller(nifty_returns)
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")

# If the series is non-stationary, apply differencing (d=1)
if adf_result[1] > 0.05:
    print("Series is non-stationary. Differencing applied.")
    nifty_returns_diff = np.diff(nifty_returns)
else:
    nifty_returns_diff = nifty_returns  # No differencing needed

# Step 2: Plot ACF and PACF for initial p, d, q estimation
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plot_acf(nifty_returns_diff, lags=15, ax=plt.gca())
plt.title("ACF - Autocorrelation Function")

plt.subplot(1, 2, 2)
plot_pacf(nifty_returns_diff, lags=15, ax=plt.gca())
plt.title("PACF - Partial Autocorrelation Function")

plt.tight_layout()
plt.show()

# Step 3: Automatically select ARIMA order (p, d, q) using AIC/BIC
best_aic = np.inf
best_order = None
best_model = None

for p in range(0, 3):  # Try different values of p
    for d in range(0, 2):  # Try different values of d
        for q in range(0, 3):  # Try different values of q
            try:
                model = ARIMA(nifty_returns, order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic
                print(f"ARIMA({p}, {d}, {q}) - AIC: {aic}")
                
                # Update best model based on AIC
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = model_fit
            except Exception as e:
                continue

print(f"Best ARIMA model order: {best_order} with AIC: {best_aic}")

# Step 4: Use ARIMA model if seasonality is suspected
# Assuming quarterly data, seasonal period is 4 (4 quarters in a year)
arima_model = SARIMAX(nifty_returns,
                       order=(best_order[0], best_order[1], best_order[2]), 
                       seasonal_order=(1, 1, 1, 4))  # Seasonality assumed every 4 quarters
arima_result = arima_model.fit()

# Print summary of the ARIMA model
print(arima_result.summary())

# Step 5: Generate forecasts and plot fitted values vs actual values
fitted_values = arima_result.fittedvalues
forecast_values = arima_result.forecast(steps=8)  # Forecast next 2 years (8 quarters)

# Create year labels for every 4th data point (quarters -> years)
years = list(range(2016, 2024))  # Assuming data starts from 2016
x_ticks = np.arange(0, len(nifty_returns), 4)  # Tick positions corresponding to yearly intervals

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(nifty_returns, label="Actual Nifty Return (%)", color="blue")
plt.plot(fitted_values, label="Fitted Nifty Return (%) (ARIMA)", color="red")
plt.plot(np.arange(len(nifty_returns), len(nifty_returns) + 8), forecast_values, label="Forecasted Nifty Return (%)", color="green")
plt.title("Nifty Quarterly Returns and ARIMA Model")
plt.xlabel("Quarters")
plt.ylabel("Nifty Return (%)")
plt.xticks(ticks=x_ticks, labels=years, rotation=45)  # Set year labels for every 4th data point
plt.legend()
plt.show()

# Step 6: Residual diagnostics (checking if residuals are white noise)
residuals = arima_result.resid
plt.figure(figsize=(12, 6))
plt.plot(residuals, label="Residuals", color="purple")
plt.title("Residuals of ARIMA Model")
plt.xlabel("Quarters")
plt.ylabel("Residuals")
plt.legend()
plt.show()

# ACF of residuals (should look like white noise)
plot_acf(residuals, lags=15)
plt.title("ACF of Residuals")
plt.show()

