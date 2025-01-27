import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

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

# Alternatively, aggregate duplicates by averaging
# data = data.groupby(data.index).mean()

# Set the frequency to hourly
data = data.asfreq('h')

# Fill missing values
data['value'] = data['value'].interpolate()

# Train the ARIMA model using the last 30 days of data (30*24 = 720 hours)
train_data = data['value'][-7*24:]  # Last 30 days of hourly data
print("Training Data Preview:")
print(train_data.head())

# Ensure training data is not empty
if train_data.empty:
    raise ValueError("Training data is empty. Ensure the dataset contains valid data.")

# Fit the ARIMA model
model = ARIMA(train_data, order=(5, 1, 0))  # Modify (p, d, q) order as needed
model_fit = model.fit()

# Forecast the next 24 hours
forecast = model_fit.forecast(steps=24)

# Prepare a DataFrame for the forecast
forecast_index = pd.date_range(start=train_data.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')
forecast_df = pd.DataFrame({'forecast': forecast}, index=forecast_index)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data['value'][-1*24:], label='Historical Prices (Last day)')
plt.plot(forecast_df, label='Forecast (Next 24 hours)', color='orange')
plt.xlabel('Time')
plt.ylabel('Electricity Price')
plt.title('Electricity Price Forecast')
plt.legend()
plt.show()

# Save the forecast to a CSV file
forecast_df.to_csv('forecast_next_24_hours.csv')
print("Forecast completed and saved to 'forecast_next_24_hours.csv'.")