import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# Load the datasets
gdp_data = pd.read_csv("../Data/gdp_data.csv", index_col="Year")
nifty_data = pd.read_csv("../Data/nifty_data.csv", index_col="Year")

# Flatten GDP and NIFTY data into time-series
gdp_values = gdp_data.values.flatten()
nifty_values = nifty_data.values.flatten()

# Compute GDP growth rate (quarter-on-quarter percentage change)
gdp_percent_change = 100 * np.diff(gdp_values) / gdp_values[:-1]

nifty_percent_change = nifty_values[1:]  # Skip the first value to match GDP diff length

# Ensure both series are of the same length
assert len(gdp_percent_change) == len(nifty_percent_change), "Mismatch in data lengths!"

# Combine data into a DataFrame for analysis
data = pd.DataFrame({
    "GDP_Percent_Change": gdp_percent_change,
    "NIFTY_Percent_Change": nifty_percent_change
})

# Check stationarity for both series
def check_stationarity(series, name):
    result = adfuller(series)
    print(f"ADF Statistic for {name}: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] > 0.05:
        print(f"{name} is non-stationary. Differencing is required.")
    else:
        print(f"{name} is stationary.")

check_stationarity(data["GDP_Percent_Change"], "GDP Percent Change")
check_stationarity(data["NIFTY_Percent_Change"], "NIFTY Percent Change")

# Fit VAR model
var_model = VAR(data)
lag_order = var_model.select_order(maxlags=8).selected_orders["aic"]  # Select optimal lag using AIC
print(f"Optimal lag order: {lag_order}")
var_result = var_model.fit(lag_order)
print(var_result.summary())

# Granger causality tests
print("\nGranger Causality Tests:")
granger_results = grangercausalitytests(data, maxlag=lag_order, verbose=True)

# Plot GDP and NIFTY percent changes over time
plt.figure(figsize=(12, 6))
x_ticks = np.arange(0, len(data), 4)
years = list(range(2016, 2024))

plt.plot(data["GDP_Percent_Change"], label="GDP Percent Change", color="blue", marker='o')
plt.plot(data["NIFTY_Percent_Change"], label="NIFTY Percent Change", color="orange", marker='o')
plt.xticks(ticks=x_ticks, labels=years, rotation=45)
plt.title("GDP and NIFTY Percent Changes (2016–2023)")
plt.xlabel("Year")
plt.ylabel("Percent Change (%)")
plt.legend()
plt.grid()
plt.show()

# Cross-correlation analysis
from statsmodels.tsa.stattools import ccf
cross_corr = ccf(data["GDP_Percent_Change"], data["NIFTY_Percent_Change"])

plt.figure(figsize=(12, 4))
plt.stem(range(len(cross_corr)), cross_corr)
plt.title("Cross-Correlation: GDP vs NIFTY Percent Changes")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid()
plt.show()

# Save the results to a CSV file
granger_results_df = pd.DataFrame()

for lag, test_result in granger_results.items():  # Iterate over lag values
    for test_name, result in test_result[0].items():  # Iterate over tests (like ssr F-test, etc.)
        try:
            # Extract p-values (if the structure is as expected)
            if isinstance(result, dict):  # Check if result is a dictionary
                p_values = [result[i][1] for i in range(lag)]
            else:
                # Handle scalar or unexpected structures
                p_values = [result[1]] if hasattr(result, '__getitem__') else [result]
                
            granger_results_df[f"{lag}_{test_name}"] = p_values
        except Exception as e:
            print(f"Error processing {lag}, {test_name}: {e}")

# Save the results to a CSV file
granger_results_df.to_csv("granger_results.csv", index=False)
print("Granger results saved to 'granger_results.csv'.")