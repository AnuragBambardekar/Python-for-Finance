import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd

"""
Our analysis indeed confirms the fact from the infographic that the top 20 stocks in the S&P500 index accounted for a significant portion of the index’s overall return.

By knowing which stocks are contributing the most to the index’s return, investors can make more informed investment decisions and adjust their portfolios accordingly. This is a key information from the risk management perspective.
"""

url = 'https://www.slickcharts.com/sp500'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0;Win64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})
headers = [header.text for header in table.find_all('th')]

data = []
for row in table.find_all('tr'):
    data.append([cell.text for cell in row.find_all('td')])

df = pd.DataFrame(data, columns=headers)

df = df.drop(0)

df = df.drop(index=df.index[20:])

df.loc[6, 'Symbol'] = 'BRK-B'

top_20 = df['Symbol'][:20].to_list()

prices = yf.download(top_20, start='2023-01-01', end='2023-04-21')['Close']

return_df = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]).to_frame().reset_index().rename(columns={'index':'Symbol', 0:'returns'})

merged = pd.merge(df, return_df, on='Symbol')

merged['Weight'] = df['Weight'].astype(float)

merged['Weighted Return'] = merged['Weight'] * merged['returns']

total_return = merged['Weighted Return'].sum()

sp500 = yf.download('^GSPC', start='2023-01-01', end='2023-04-21')['Close']

sp500_return = ((sp500.iloc[-1] - sp500.iloc[0]) / sp500.iloc[0]) *100

percent_of_sp500_return = (total_return / sp500_return) * 100

print(percent_of_sp500_return)