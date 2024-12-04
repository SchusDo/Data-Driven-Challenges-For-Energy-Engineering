import pandas as pd

# Load the raw data
file_path = "MIBGAS_Data_2024.csv"  # file name

# Skip the first line containing metadata
data = pd.read_csv(file_path, sep=';', skiprows=1)

# Clean up column names
data.columns = data.columns.str.strip()

# Filter for GDAES_D+1 product
filtered_data = data[data['Product'] == 'GDAES_D+1']

# Select necessary columns
filtered_data = filtered_data[['Trading day', 'MIBGAS Daily Price [EUR/MWh]']]

# Rename columns for clarity
filtered_data.rename(
    columns={'Trading day': 'datetime_utc', 'MIBGAS Daily Price [EUR/MWh]': 'fuel_price'},
    inplace=True
)

# Convert Trading day to datetime and set as index
filtered_data['datetime_utc'] = pd.to_datetime(filtered_data['datetime_utc'], format='%d/%m/%Y')
filtered_data.set_index('datetime_utc', inplace=True)

# Sort by date
filtered_data.sort_index(inplace=True)

# Save the cleaned data to a new CSV
output_file = "fuel_price.csv"
filtered_data.to_csv(output_file)

print(f"Fuel price data prepared and saved to '{output_file}'.")
