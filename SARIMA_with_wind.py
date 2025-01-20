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


# Load the wind forecast
wind_data = pd.read_csv(
    'datas_v2/WIND_D+1_DAILY_FORECAST.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)

# Remove datetime and leave only datetime UTC
wind_data = wind_data.drop(wind_data.columns[0], axis=1)

# Show the last data available
print('Last available wind date is', wind_data.index[-1])
print('Last available electricity price date is', electricity_data.index[-1])


# Filter for Spain (geo_name = 'España') or Península, if necessary
electricity_data = electricity_data[electricity_data['geo_name'] == 'España']
wind_data = wind_data[wind_data['geo_name'] == 'Península']

# Ensure both datasets are sorted and align their indices
electricity_data = electricity_data.sort_index()
wind_data = wind_data.sort_index()
#print(wind_data.head())

# Handle duplicate timestamps in electricity data
if electricity_data.index.duplicated().sum() > 0:
    print(f"Found {electricity_data.index.duplicated().sum()} duplicate timestamps. Removing duplicates...")
    electricity_data = electricity_data[~electricity_data.index.duplicated(keep='first')]

# Handle duplicate timestamps in wind data
if wind_data.index.duplicated().sum() > 0:
    print(f"Found {wind_data.index.duplicated().sum()} duplicate timestamps. Removing duplicates...")
    wind_data = wind_data[~wind_data.index.duplicated(keep='first')]


# Set the frequency to hourly for electricity data
electricity_data = electricity_data.asfreq('h')
wind_data = wind_data.asfreq('h')

# Forward-fill daily fuel price to align with hourly electricity data
wind_data = wind_data.reindex(electricity_data.index, method='ffill')

# Interpolate missing electricity prices
#electricity_data['value'] = electricity_data['value'].interpolate()
#wind_data['value'] = wind_data['value'].interpolate()


# Check for NaN values
if electricity_data.isna().sum().sum() > 0:
    raise ValueError("Data contains NaN values after merging. Please ensure all missing values are handled.")

# Define dependent and exogenous variables directly
y_train = electricity_data['value'][:-24]
y_test = electricity_data['value'][-24:]
X_train = wind_data['value'][:-24]
X_test = wind_data['value'][-24:]

# Fit SARIMAX
model = SARIMAX(y_train, exog=X_train, order=(1, 1, 0), seasonal_order=(1, 1, 1, 24))
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
plt.title('Electricity Price Forecast with SARIMA and Wind forecast')
plt.legend()
plt.show()

# Save the forecast and actual values to a CSV file for further analysis
forecast_df.to_csv('forecast_vs_actual_with_wind.csv')
print("Forecast vs actual results saved to 'forecast_vs_actual_with_wind.csv'.")
