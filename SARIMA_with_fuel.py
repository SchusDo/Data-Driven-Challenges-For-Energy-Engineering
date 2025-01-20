import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load the electricity price data
electricity_data = pd.read_csv(
    'historical_spainPrice.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)

# Load the fuel price data
fuel_data = pd.read_csv(
    'datas_v2/fuel_price.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)

fuel_data.index = pd.to_datetime(fuel_data.index).tz_localize('UTC')

# Filter for Spain (geo_name = 'España'), if necessary
electricity_data = electricity_data[electricity_data['geo_name'] == 'España']

# Ensure both datasets are sorted and align their indices
electricity_data = electricity_data.sort_index()
fuel_data = fuel_data.sort_index()
print(fuel_data.head())

# Handle duplicate timestamps in electricity data
if electricity_data.index.duplicated().sum() > 0:
    print(f"Found {electricity_data.index.duplicated().sum()} duplicate timestamps. Removing duplicates...")
    electricity_data = electricity_data[~electricity_data.index.duplicated(keep='first')]

# Set the frequency to hourly for electricity data
electricity_data = electricity_data.asfreq('h')
#fuel_data = fuel_data.asfreq('D')

# Forward-fill daily fuel price to align with hourly electricity data
fuel_data = fuel_data.reindex(electricity_data.index, method='ffill')
print(fuel_data.head())

# Interpolate missing electricity prices
electricity_data['value'] = electricity_data['value'].interpolate()

# Combine the datasets
electricity_data['fuel_price'] = fuel_data['fuel_price']

# Check for NaN values
if electricity_data.isna().sum().sum() > 0:
    raise ValueError("Data contains NaN values after merging. Please ensure all missing values are handled.")

# Split the data into training and testing sets
train_data = electricity_data[:-24]  # All but the last 24 hours for training
test_data = electricity_data[-24:]  # Last 24 hours for testing

# Define the dependent variable and exogenous variable
y_train = train_data['value']
X_train = train_data[['fuel_price']]
y_test = test_data['value']
X_test = test_data[['fuel_price']]

# Define seasonal order for SARIMA
seasonal_order = (1, 1, 1, 24)  # Adjust these values based on seasonality

# Fit the SARIMA model with exogenous variable
model = SARIMAX(y_train, exog=X_train, order=(1, 1, 0), seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

# Forecast the next 24 hours
forecast = model_fit.forecast(steps=24, exog=X_test)

# Calculate accuracy metrics
mae = mean_absolute_error(y_test, forecast)
rmse = np.sqrt(mean_squared_error(y_test, forecast))
mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Prepare a DataFrame for the forecast
forecast_df = pd.DataFrame({'actual': y_test, 'forecast': forecast}, index=y_test.index)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_train[-2*24:], label='Training Data (Last 2 days)')
plt.plot(y_test, label='Actual Prices (Last 24 hours)', color='blue', linestyle='--')
plt.plot(forecast_df['forecast'], label='Forecast (Next 24 hours)', color='orange')
plt.xlabel('Time')
plt.ylabel('Electricity Price')
plt.title('Electricity Price Forecast with SARIMA and Fuel Price')
plt.legend()
plt.show()

# Save the forecast and actual values to a CSV file for further analysis
forecast_df.to_csv('forecast_vs_actual_with_fuel.csv')
print("Forecast vs actual results saved to 'forecast_vs_actual_with_fuel.csv'.")
