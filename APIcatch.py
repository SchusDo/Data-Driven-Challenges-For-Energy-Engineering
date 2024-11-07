import requests
import pandas as pd
import numpy as np


API_TOKEN = 'fb4b803d6e48a2f0e5b4f2cdbc6cf3811abe64ce6ee763bf71ac7bdf2f4a39c1'

headers = {
    'Host': 'api.esios.ree.es',
    'x-api-key': API_TOKEN
}

URL_BASE = 'https://api.esios.ree.es/'
ENDPOINT = 'indicators/'
INDICATOR = '600' 

url = URL_BASE + ENDPOINT + INDICATOR


params = {
    'start_date': '2023-01-01T01',
    'end_date': '2023-02-01T00'
}


res = requests.get(url, headers=headers, params=params)
data = res.json()
#print(data)

df = pd.DataFrame(data['indicator']['values'])
df = df[['datetime_utc', 'geo_id', 'geo_name', 'value']]

#print(df)

df_españa = df[df['geo_name'] == 'España']
df_españa.reset_index(drop=True, inplace=True)

print(df_españa)

# save file to a csv
df_españa.to_csv('spainPrice2023.csv', index=False, encoding='utf-8')

