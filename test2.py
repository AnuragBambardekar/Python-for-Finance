import yfinance as yf
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from bokeh.embed import components
from bokeh.plotting import figure
from datetime import *
from bokeh.models import ColumnDataSource
sns.set()
from yahoo_fin import stock_info as si

sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
data_table = pd.read_html(sp500url)[0]

sp500_symbols = data_table["Symbol"].to_list()

end_date = date.today()
start_date = end_date - pd.DateOffset(365*5) # 5 year data

stocks_list = si.tickers_sp500() # all s&p500 stocks
df_sp500 = yf.download(tickers = stocks_list, start = start_date, end = end_date)
sp500_symbols = df_sp500["Close"].columns.to_list()
df = df_sp500.sort_index()

gap_returns = np.log(df["Open"]/df["Close"].shift(1))
intraday_returns = np.log(df["Close"]/df["Open"])
df_variation =  df["Adj Close"].pct_change()
df_volatility=df_variation.rolling(250).std()*100*np.sqrt(250)

weekday = gap_returns.index.map(lambda x: x.weekday())

best_day=pd.concat([
    gap_returns.groupby(weekday).mean().T.mean().rename("Gap_return mean"),
    gap_returns.groupby(weekday).std().T.mean().rename("Gap_return std"),
    
    intraday_returns.groupby(weekday).mean().T.mean().rename("IntraDay_return mean"),
    intraday_returns.groupby(weekday).std().T.mean().rename("IntraDay_return std"),
    
    df_volatility.groupby(weekday).mean().T.mean().rename("Volatility"),
],axis=1)

best_day.reset_index(inplace=True)
best_day["Date"] = best_day["Date"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"})
best_day.rename(columns={"Date":"Day"},inplace=True)

df_perCompany=pd.DataFrame( data_table[['Symbol', 'GICS Sector']])
df_perCompany.rename(columns={"Symbol":"Ticker"},inplace=True)

for ticker in sp500_symbols:
    df_adjClose_ticker=df["Adj Close"][ticker].dropna()
    if df_adjClose_ticker.shape[0]==0:
        continue
    year_index = df_adjClose_ticker.index.map(lambda x: x.year)

    first_close, last_close = df_adjClose_ticker.iloc[[0,-1]]
    total_return = (last_close/first_close)-1
    first_year = df_adjClose_ticker.index[0].year
    last_year = df_adjClose_ticker.index[-1].year

    years=last_year-first_year+1
    returnPerYear=[]
    for year in range(first_year,last_year+1):
        first_close_year, last_close_year = df_adjClose_ticker[year_index==year].iloc[[0,-1]]
        year_return= (last_close_year/first_close_year)-1
        returnPerYear.append(year_return)
    mean_return_per_year = np.mean(returnPerYear)
    volatility = np.std(returnPerYear)
    df_perCompany.loc[df_perCompany["Ticker"]==ticker,["years","total_return","mean_return_per_year","volatility"]]=years,total_return,mean_return_per_year,volatility
    
df_perCompany.dropna(inplace=True)

Rf = 0.01/255
df_perCompany["Return_Volatility_Ratio"] = (df_perCompany["mean_return_per_year"]*df_perCompany["total_return"])/((df_perCompany["volatility"]-Rf)*df_perCompany["years"])
top10_companies=df_perCompany.sort_values(by="Return_Volatility_Ratio",ascending=False)[0:10]
# tickers_list = top10_companies['Ticker'].to_list()
# sector_list = top10_companies['GICS Sector'].to_list()
# Return_Volatility_Ratio = top10_companies['Return_Volatility_Ratio'].to_list()

df_perSector=df_perCompany.groupby("GICS Sector").mean()
Rf = 0.01/255
df_perSector["Return_Volatility_Ratio"] = (df_perSector["mean_return_per_year"]*df_perSector["total_return"])/((df_perSector["volatility"]-Rf)*df_perSector["years"])
df_perSector.sort_values("Return_Volatility_Ratio",ascending=False,inplace=True)

min_ratio=df_perCompany["total_return"].min()
max_ratio=df_perCompany["total_return"].max()
total_return_scale = (df_perCompany["total_return"]+1-min_ratio)/(max_ratio-min_ratio)

from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FactorRange

# Create a ColumnDataSource for the sector data
source = ColumnDataSource(df_perSector)

# Create the figure
p = figure(x_range=FactorRange(factors=df_perSector.index.tolist()), 
           height=500, 
           width=1000,
           title='Return Volatility Ratio by Sector')

# Add the vertical bars
p.vbar(x='GICS Sector', 
       top='Return_Volatility_Ratio', 
       width=0.9, 
       source=source, 
       line_color='white')

# Set the axis labels
p.xaxis.axis_label = "Sector"
p.yaxis.axis_label = "Return Volatility Ratio"

# Show the figure
show(p)
