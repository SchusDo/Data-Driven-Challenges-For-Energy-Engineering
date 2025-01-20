# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:03:44 2024

@author: ZEPHYRUS
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

API_TOKEN = 'XXXXXXXXX'
# contact ESIOS to get a valid token: consultasios@ree.es

headers = {
    'Host': 'api.esios.ree.es',
    'x-api-key': API_TOKEN
}

URL_BASE = 'https://api.esios.ree.es/'
ENDPOINT = 'indicators/'
INDICATOR = '600' 
# Daily SPOT market price

url = URL_BASE + ENDPOINT + INDICATOR

# Define the start date and the current date
end_date = datetime.now()
start_date = start_date = datetime(2024, 1, 1)#(2015, 1, 1)

# Loop over each month in the past 10 years
current_date = start_date
csv_files = []

while current_date < end_date:
    # Set start and end dates for each month
    start_date_str = current_date.strftime('%Y-%m-%dT%H')
    next_month = current_date + timedelta(days=31)  # Move roughly to the next month
    next_month = next_month.replace(day=1)  # Set to the first day of the next month
    end_date_str = next_month.strftime('%Y-%m-%dT%H')

    params = {
        'start_date': start_date_str,
        'end_date': end_date_str
    }

    # Request data for the month
    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200:
        data = res.json()
        
        if 'indicator' in data and 'values' in data['indicator']:
            df = pd.DataFrame(data['indicator']['values'])
            df = df[['datetime_utc', 'geo_id', 'geo_name', 'value']]
            
            # Filter for Spain and reset index
            df_spain = df[df['geo_name'] == 'España']
            df_spain.reset_index(drop=True, inplace=True)

            # Save the monthly data to CSV
            month_filename = f'spainPrice_{current_date.strftime("%Y_%m")}.csv'
            df_spain.to_csv(month_filename, index=False, encoding='utf-8')
            csv_files.append(month_filename)
            print(f"Saved {month_filename}")
        else:
            print(f"No data found for {start_date_str} to {end_date_str}")
    else:
        print(f"Error: {res.status_code} for {start_date_str} to {end_date_str}")

    # Move to the next month
    current_date = next_month

# Combine all monthly CSV files into a single historical file
historical_data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
historical_data.to_csv('historical_spainPrice.csv', index=False, encoding='utf-8')
print("Combined all monthly data into historical_spainPrice.csv")
