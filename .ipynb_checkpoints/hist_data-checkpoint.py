import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import requests
sns.set_style('darkgrid')

access_token='eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3WUIzRTkiLCJqdGkiOiI2N2JiZWFiNWE3NmQyMjZlYTRlMTYzMzMiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaWF0IjoxNzQwMzY4NTY1LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NDA0MzQ0MDB9.FW25xkrnMuRuenoicXFm_OOquDf5W32xz7oJleOYiUs'


def get_historical_data(duration,start_date,end_date,access_token):
  url = f'https://api.upstox.com/v2/historical-candle/BSE_EQ|INE092E01011/{duration}/{start_date}/{end_date}'
  headers = {
      "accept": "application/json",
      "Authorization": f"Bearer {access_token}",
      "Content-Type": "application/json"
  }
  response = requests.get(url, headers=headers)

  # Check the response status
  if response.status_code == 200:
      # Do something with the response data (e.g., print it)
      print(response.json())
  else:
      # Print an error message if the request was not successful
      print(f"Error: {response.status_code} - {response.text}")

  return response


duration='1minute'
start_date='2025-02-14'
end_date='2023-01-01'

response6=get_historical_data(duration,start_date,end_date,access_token).json()

columns=['timestamp','open','high','low','close','volume','oi']
hist_data = pd.DataFrame(response6['data']['candles'],columns=columns)

print(hist_data.head())

hist_data.to_csv('MarketData.csv')