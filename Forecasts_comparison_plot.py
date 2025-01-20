import pandas as pd
import matplotlib.pyplot as plt

# Load the the electricity data 
electricity_data = pd.read_csv(
    'historical_spainPrice.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)

#Load forecasts
forecast_p48 =pd.read_csv(
    'forecast_vs_actual_with_p48.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)
forecast_wind =pd.read_csv(
    'forecast_vs_actual_with_wind.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)
forecast_fuel =pd.read_csv(
    'forecast_vs_actual_with_fuel.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)
forecast_sarima =pd.read_csv(
    'forecast_vs_actual.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)

# Ensure both datasets are sorted and align their indices
electricity_data = electricity_data.sort_index()

# Handle duplicate timestamps in electricity data
if electricity_data.index.duplicated().sum() > 0:
    print(f"Found {electricity_data.index.duplicated().sum()} duplicate electricity timestamps. Removing duplicates...")
    electricity_data = electricity_data[~electricity_data.index.duplicated(keep='first')]

electricity_data = electricity_data.asfreq('h')

window=24
y_train = electricity_data['value'][:-window]
y_test = electricity_data['value'][-window:]



# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_train[-2*24:], label='Training Data (Last 2 days)')
plt.plot(y_test, label='Actual Prices (Last 24 hours)', color='blue', linestyle='--')
plt.plot(forecast_p48['forecast'], label='Forecast w/ P48')
plt.plot(forecast_wind['forecast'], label='Forecast w/ wind')
plt.plot(forecast_fuel['forecast'], label='Forecast w/ fuel')
plt.plot(forecast_sarima['forecast'], label='Forecast SARIMA')
plt.xlabel('Time')
plt.ylabel('Electricity Price')
plt.title('Electricity Price Forecasts with SARIMA and Exogenous Variables')
plt.legend()
plt.show()

