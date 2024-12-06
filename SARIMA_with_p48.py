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
    'WIND_D+1_DAILY_FORECAST.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)

# Load the PV forecast
pv_data = pd.read_csv(
    'PHOTOVOLTAIC_D+1_DAILY_FORECAST.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)

# Load the solar thermal forecast
solar_data = pd.read_csv(
    'SOLAR_THERMAL_FORECAST.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)

# Load the demand forecast
demand_data = pd.read_csv(
    'DEMAND_D+1_DAILY_FORECAST.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)

# Remove datetime and leave only datetime UTC
wind_data = wind_data.drop(wind_data.columns[0], axis=1)
pv_data = pv_data.drop(pv_data.columns[0], axis=1)
solar_data = solar_data.drop(solar_data.columns[0], axis=1)
demand_data = demand_data.drop(demand_data.columns[0], axis=1)

# Show the last data available
print('Last available wind date is', wind_data.index[-1])
print('Last available electricity price date is', electricity_data.index[-1])


# Filter for Spain (geo_name = 'España') or Península, if necessary
electricity_data = electricity_data[electricity_data['geo_name'] == 'España']
wind_data = wind_data[wind_data['geo_name'] == 'Península']
pv_data = pv_data[pv_data['geo_name'] == 'Península']
solar_data = solar_data[solar_data['geo_name'] == 'Península']
demand_data = demand_data[demand_data['geo_name'] == 'Península']


# Ensure both datasets are sorted and align their indices
electricity_data = electricity_data.sort_index()
wind_data = wind_data.sort_index()
pv_data = pv_data.sort_index()
solar_data = solar_data.sort_index()
demand_data = demand_data.sort_index()

#print(wind_data.head())

# Handle duplicate timestamps in electricity data
if electricity_data.index.duplicated().sum() > 0:
    print(f"Found {electricity_data.index.duplicated().sum()} duplicate electricity timestamps. Removing duplicates...")
    electricity_data = electricity_data[~electricity_data.index.duplicated(keep='first')]

# Handle duplicate timestamps in wind data
if wind_data.index.duplicated().sum() > 0:
    print(f"Found {wind_data.index.duplicated().sum()} duplicate wind timestamps. Removing duplicates...")
    wind_data = wind_data[~wind_data.index.duplicated(keep='first')]

# Handle duplicate timestamps in PV data
if pv_data.index.duplicated().sum() > 0:
    print(f"Found {pv_data.index.duplicated().sum()} duplicate PV timestamps. Removing duplicates...")
    pv_data = pv_data[~pv_data.index.duplicated(keep='first')]

# Handle duplicate timestamps in Solat Thermal data
if solar_data.index.duplicated().sum() > 0:
    print(f"Found {solar_data.index.duplicated().sum()} duplicate solar thermal timestamps. Removing duplicates...")
    solar_data = solar_data[~solar_data.index.duplicated(keep='first')]

# Handle duplicate timestamps in demand data
if demand_data.index.duplicated().sum() > 0:
    print(f"Found {demand_data.index.duplicated().sum()} duplicate demand timestamps. Removing duplicates...")
    demand_data = demand_data[~demand_data.index.duplicated(keep='first')]


# Set the frequency to hourly for electricity data. 
electricity_data = electricity_data.asfreq('h')
wind_data = wind_data.asfreq('h')
pv_data = pv_data.asfreq('h')
solar_data = solar_data.asfreq('h')
demand_data = demand_data.asfreq('h')


# Forward-fill daily fuel price to align with hourly electricity data
# This will only happen if the P48 are not up to date.
wind_data = wind_data.reindex(electricity_data.index, method='ffill')
pv_data = pv_data.reindex(electricity_data.index, method='ffill')
solar_data = solar_data.reindex(electricity_data.index, method='ffill')
demand_data = demand_data.reindex(electricity_data.index, method='ffill')

# Interpolate missing electricity prices
#electricity_data['value'] = electricity_data['value'].interpolate()
#wind_data['value'] = wind_data['value'].interpolate()


# Check for NaN values
#if electricity_data.isna().sum().sum() > 0:
#    raise ValueError("Data contains NaN values after merging. Please ensure all missing values are handled.")

# Define dependent and exogenous variables directly
y_train = electricity_data['value'][:-24]
y_test = electricity_data['value'][-24:]

# Exogenous variables (wind,pv, solar) Built in a dataframe
X_train = pd.DataFrame({
    'wind':wind_data['value'][:-24],
    'pv':pv_data['value'][:-24],
    'solar':solar_data['value'][:-24],
    'demand':demand_data['value'][:-24]
})

X_test = pd.DataFrame({
    'wind':wind_data['value'][-24:],
    'pv':pv_data['value'][-24:],
    'solar':solar_data['value'][-24:],
    'demand':demand_data['value'][-24:]
})


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
plt.title('Electricity Price Forecast with SARIMA and P48 values')
plt.legend()
plt.show()

# Save the forecast and actual values to a CSV file for further analysis
forecast_df.to_csv('forecast_vs_actual_with_p48.csv')
print("Forecast vs actual results saved to 'forecast_vs_actual_with_p48.csv'.")
