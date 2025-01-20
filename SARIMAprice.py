import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
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

# Split the data into training and testing sets
train_data = data['value'][:-24]  # All but the last 24 hours for training
test_data = data['value'][-24:]  # Last 24 hours for testing
print("Training Data Preview:")
print(train_data.head())
print(train_data.tail())
print(test_data.head())

# Ensure training data is not empty
if train_data.empty or test_data.empty:
    raise ValueError("Training or testing data is empty. Ensure the dataset contains valid data.")

# Define seasonal order for SARIMA
seasonal_order = (1, 1, 1, 24)  # P,Q,D,s Adjust these values based on seasonality

# Fit the SARIMA model
model = SARIMAX(train_data, order=(1, 1, 0), seasonal_order=seasonal_order)
model_fit = model.fit(disp=False)

# Forecast the next 24 hours
forecast = model_fit.forecast(steps=24)

# Calculate accuracy metrics
mae = mean_absolute_error(test_data, forecast)
rmse = np.sqrt(mean_squared_error(test_data, forecast))
mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Prepare a DataFrame for the forecast
forecast_df = pd.DataFrame({'actual': test_data, 'forecast': forecast}, index=test_data.index)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train_data[-2*24:], label='Training Data (Last 2 days)')
plt.plot(test_data, label='Actual Prices (Last 24 hours)', color='blue', linestyle='--')
plt.plot(forecast_df['forecast'], label='Forecast (Next 24 hours)', color='orange')
plt.xlabel('Time')
plt.ylabel('Electricity Price')
plt.title('Electricity Price Forecast with SARIMA')
plt.legend()
plt.show()

# Save the forecast and actual values to a CSV file for further analysis
forecast_df.to_csv('forecast_vs_actual.csv')
print("Forecast vs actual results saved to 'forecast_vs_actual.csv'.")