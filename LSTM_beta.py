# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:38:18 2024

@author: ZEPHYRUS
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load historical data
data = pd.read_csv('historical_spainPrice.csv', parse_dates=['datetime_utc'])
data.set_index('datetime_utc', inplace=True)

# Sort the data (just in case)
data.sort_index(inplace=True)

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data['value'], label='Electricity Price')
plt.title('Electricity Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['value'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + forecast_horizon])  # Next 24 hours
    return np.array(X), np.array(y)

SEQ_LENGTH = 30 * 24  # 30 days of hourly data (30 * 24)
FORECAST_HORIZON = 24  # Predict the next 24 hours
X, y = create_sequences(scaled_data, SEQ_LENGTH, FORECAST_HORIZON)

# Split into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(FORECAST_HORIZON)
])


# Hpyerparameter optimization is missing
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model, batch size original: 32, epochs = 4
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4, batch_size=10, verbose=1)
print('plot7')

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
print('plot8')

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Undo scaling
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, FORECAST_HORIZON))
print('plot9')

# Plot predictions vs. actual
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[0], label='Actual Prices (Next 24 hours)')
plt.plot(predictions[0], label='Predicted Prices (Next 24 hours)')
plt.title('Electricity Price Prediction for the Next Day')
plt.xlabel('Hour')
plt.ylabel('Price')
plt.legend()
plt.show()
print('plot10')

# Predict the next day's price
last_sequence = scaled_data[-SEQ_LENGTH:]  # Last 30 days of hourly data
last_sequence = np.expand_dims(last_sequence, axis=0)  # Reshape for LSTM
next_day_scaled = model.predict(last_sequence)
next_day_prices = scaler.inverse_transform(next_day_scaled)
print(f"Predicted electricity prices for the next 24 hours: {next_day_prices[0]}")

# not showing the 24 hours!