# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:38:17 2024

@author: Dominik.Schuster
"""



from datetime import datetime
import requests

# Function to parse the date with optional microseconds and timezone
def parse_date(date_string):
    try:
        # Handle UTC timestamps ending in 'Z'
        if date_string.endswith('Z'):
            return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')
        # Handle timestamps with a timezone offset (like +02:00)
        elif '+' in date_string or '-' in date_string:
            # Remove the colon in the timezone if present and parse
            date_string = date_string[:-3] + date_string[-2:]
            return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%f%z')
    except ValueError as e:
        print(f"Error parsing date {date_string}: {e}")
        return None

# API endpoint and headers
url = "https://api.esios.ree.es/indicators/1001"
headers = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": "96c56fcd69dd5c29f569ab3ea9298b37151a1ee488a1830d353babad3ec90fd7"
}

# Make the GET request
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Use a set to store unique timestamp and price combinations
    seen_records = set()

    # Extract the values and their timestamps
    values = data.get('indicator', {}).get('values', [])
    for value in values:
        timestamp = value.get('datetime_utc', '')
        price = value.get('value', '')

        # Convert and format the timestamp
        parsed_date = parse_date(timestamp)
        if parsed_date:
            formatted_date = parsed_date.strftime('%Y-%m-%d %H:%M:%S')

            # Only print unique (date, price) pairs
            record = (formatted_date, price)
            if record not in seen_records:
                seen_records.add(record)
                print(f"Date: {formatted_date}, Price: {price}")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
