import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the historical data
data = pd.read_csv(
    'historical_spainPrice.csv',
    parse_dates=['datetime_utc'],  # Parse datetime column
    index_col='datetime_utc'      # Set datetime_utc as the index
)

# Filter for Spain (geo_name = 'España'), if necessary
data = data[data['geo_name'] == 'España']

# Ensure the index is sorted
data = data.sort_index()

# Handle duplicate timestamps
if data.index.duplicated().sum() > 0:
    print(f"Found {data.index.duplicated().sum()} duplicate timestamps. Removing duplicates...")
    data = data[~data.index.duplicated(keep='first')]

# Set the frequency to hourly
data = data.asfreq('h')

# Fill missing values
data['value'] = data['value'].interpolate()

# Train the ARIMA model using the last 7 days of data (7*24 = 168 hours)
train_data = data['value'][-7*24:-24]  # Last 7 days excluding the last 24 hours
test_data = data['value'][-24:]  # Last 24 hours as validation/test data

print("Training Data Preview:")
print(train_data.head())

# Ensure training data is not empty
if train_data.empty or test_data.empty:
    raise ValueError("Training or test data is empty. Ensure the dataset contains valid data.")

# Fit the ARIMA model
model = ARIMA(train_data, order=(5, 1, 0))  # Modify (p, d, q) order as needed
model_fit = model.fit()

# Forecast the next 24 hours
forecast = model_fit.forecast(steps=24)

# Prepare a DataFrame for the forecast
forecast_index = pd.date_range(start=train_data.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')
forecast_df = pd.DataFrame({'forecast': forecast}, index=forecast_index)

# Calculate Error Metrics
mae = mean_absolute_error(test_data, forecast)
rmse = np.sqrt(mean_squared_error(test_data, forecast))
mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

print("Error Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['value'][-48:], label='Historical Prices (Last 2 days)', color='blue')
plt.plot(test_data.index, test_data, label='Actual Prices (Last day)', color='green')
plt.plot(forecast_df, label='Forecast (Next 24 hours)', color='orange')
plt.xlabel('Time')
plt.ylabel('Electricity Price')
plt.title('Electricity Price Forecast')
plt.legend()
plt.show()

# Save the forecast to a CSV file
forecast_df.to_csv('forecast_next_24_hours.csv')
print("Forecast completed and saved to 'forecast_next_24_hours.csv'.")
