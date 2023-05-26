import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

#import libraries 
import yfinance as yf
import pandas as pd

#initiate the function
def betas(markets, stocks, start_date, end_date):
#download the historical data for the index/market
  market = yf.download(markets, start_date, end_date)
  market['stock_name'] = markets
#calculate daily returns 
  market['daily_return'] = market['Close'].pct_change(1)
#calculate standard deviation of the returns
  market_std = market['daily_return'].std()
  market.dropna(inplace=True)
  market = market[['Close', 'stock_name', 'daily_return']] 
#download the historical data for each stock and calculate its standard deviation 
#using for loops/iteration 
  frames = []
  stds = []
  for i in stocks: 
    data = yf.download(i, start_date, end_date)
    data['stock_name'] = i
    data['daily_return'] = data['Close'].pct_change(1)
    data.dropna(inplace=True)
    data = data[[ 'Close', 'stock_name', 'daily_return']]
    data_std = data['daily_return'].std()
    frames.append(data)
    stds.append(data_std)
#for each stock calculate its correlation with index/market 
  stock_correlation = []
  for i in frames: 
    correlation = i['daily_return'].corr(market['daily_return'])
    stock_correlation.append(correlation)
#calculate beta 
  betas = []
  for b,i in zip(stock_correlation, stds):
    beta_calc = b * (i/market_std)
    betas.append(beta_calc)
#form dataframe with the results 
  dictionary = {stocks[e]: betas[e] for e in range(len(stocks))}
  dataframe = pd.DataFrame([dictionary]).T
  dataframe.reset_index(inplace=True)
  dataframe.rename(
    columns={"index": "Stock_Name", 0: "Beta"},
    inplace=True,)
  return dataframe

companies=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
table  = companies[0]
df = table[table["Symbol"].str.contains("BRK.B|BF.B") == False]
ticker_list = df['Symbol'].to_list()
ticker_list[0:]

betas = betas('^GSPC', ticker_list, '2010-01-01', '2023-01-27')