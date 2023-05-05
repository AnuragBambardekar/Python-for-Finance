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
from bokeh.models import ColumnDataSource
from datetime import *
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

fig, axs = plt.subplots(1,2, figsize=(20,5))
sns.barplot(x=best_day["Day"],y=best_day["Gap_return mean"],ax=axs[0])
axs[0].set_title("Mean Gap Return per Day of the Week")

sns.barplot(x=best_day["Day"],y=best_day["IntraDay_return mean"],ax=axs[1])
axs[1].set_title("Mean IntraDay Return per Day of the Week")

    # convert the Seaborn plot to a Bokeh plot
p = figure(x_range=best_day["Day"], height=400, width=800, title="Mean Returns per Day of the Week")
source = ColumnDataSource(best_day)
p.vbar(x='Day', top='Gap_return mean', source=source, width=0.5, color='blue', legend_label="Gap Returns")
p.vbar(x='Day', top='IntraDay_return mean', source=source, width=0.5, color='red', legend_label="Intraday Returns")
p.xaxis.axis_label = "Day of the Week"
p.yaxis.axis_label = "Mean Return"

# generate the JavaScript and HTML code needed to embed the Bokeh plot into an HTML template
# script, div = components(p)

from bokeh.io import show

show(p)