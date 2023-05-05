import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yahoo_fin import stock_info as si
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from datetime import *

sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
data_table = pd.read_html(sp500url)
data_table[0]

end_date = date.today()
start_date = end_date - pd.DateOffset(365*5) # 5 year data

stocks_list = si.tickers_sp500() # all s&p500 stocks
Five_yr_all_prices_df = yf.download(tickers=stocks_list, start=start_date, end=end_date)

prices_df = Five_yr_all_prices_df.stack()

prices_df.to_csv('prices_df1.csv')

df = df = pd.read_csv('prices_df1.csv', parse_dates=['Date'])

df.columns = ['Date','Symbol','Adj Close','Close','High','Low','Open','Volume']
df = df[['Date','Close','Symbol']]

stocks = df.pivot_table(index='Date', columns='Symbol', values='Close')
stocks = stocks.dropna(axis=1)

stocks.index = pd.to_datetime(stocks.index, utc=True)

stocks = stocks.resample('W').last()

start = stocks.iloc[0]
returns = (stocks - start) / start

best = returns.iloc[-1].sort_values(ascending=False).head()
worst = returns.iloc[-1].sort_values().head()

def get_name(symbol):
    name = symbol
    try:
        # Convert the DataFrame to a dictionary
        symbol_to_name = dict(zip(data_table[0]['Symbol'], data_table[0]['Security']))
        # Look up the name based on the symbol
        name = symbol_to_name[symbol]
    except KeyError:
        pass
    return name

from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10_10
from bokeh.transform import factor_cmap
from bokeh.models import HoverTool

def plot_stock(symbol, stocks=stocks):
    name = get_name(symbol)
    # Create a Bokeh figure
    p = figure(title=name, x_axis_label='Date', y_axis_label='Price', x_axis_type='datetime')

    # Create a ColumnDataSource from the stocks dataframe for the given symbol
    source = ColumnDataSource(stocks[[symbol]])

    # Plot the line chart
    p.line(x='Date', y=symbol, source=source, legend_label=name, line_width=2, line_color='navy')

    # Add hover tool
    p.add_tools(HoverTool(tooltips=[('Date', '@Date{%F}'), ('Price', '@' + symbol)], formatters={'@Date': 'datetime'}))

    # Show the plot
    show(p)


names1 = pd.DataFrame({'name':[get_name(symbol) for symbol in best.index.to_list()]}, index = best.index)
best = pd.concat((best, names1), axis=1)

names2 = pd.DataFrame({'name':[get_name(symbol) for symbol in worst.index.to_list()]}, index = worst.index)
worst = pd.concat((worst, names2), axis=1)

best_first_symbol = best.index[0]
worst_first_symbol = worst.index[0]

# print(best_first_symbol)
# print(worst_first_symbol)

plot_stock(best_first_symbol, stocks=returns)
plot_stock(worst_first_symbol, stocks=returns)

kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(returns.T)

clusters = {}
for l in np.unique(kmeans.labels_):
    clusters[l] = []

for i,l in enumerate(kmeans.predict(returns.T)):
    clusters[l].append(returns.columns[i])

# Create a dictionary of the clusters
cluster_dict = {}
for c in sorted(clusters):
    cluster_dict[c] = [get_name(symbol)+' ('+symbol+')' for symbol in clusters[c]]

# Import necessary Bokeh libraries
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, show

# Create a dictionary of the clusters
cluster_dict = {}
for c in sorted(clusters):
    cluster_dict[c] = [get_name(symbol)+' ('+symbol+')' for symbol in clusters[c]]

# Create a list to hold all the plots for the clusters
all_plots = []

# Loop through each cluster and create a plot for each stock in the cluster
for c in sorted(clusters):
    # Create a list to hold all the data sources for each stock in the cluster
    sources = []
    
    # Loop through each stock in the cluster and create a data source for it
    for symbol in clusters[c]:
        name = get_name(symbol)
        source = ColumnDataSource(data=dict(x=returns.index, y=returns[symbol], name=[name]*len(returns.index), symbol=[symbol]*len(returns.index)))
        sources.append(source)
        
    # Create a Bokeh figure
    p = figure(title="Returns (Clusters from PCA components) cluster " + str(c), x_axis_label='Date', y_axis_label='Returns', x_axis_type='datetime')

    # Add a line plot for each stock in the cluster
    for i, source in enumerate(sources):
        p.line(x='x', y='y', source=source, legend_label=cluster_dict[c][i], line_width=2)

    # Add a hover tool
    hover = HoverTool(tooltips=[('Name', '@name'), ('Symbol', '@symbol'), ('Date', '@x{%F}'), ('Returns', '@y{0.2f}%')], formatters={'@x': 'datetime'})
    p.add_tools(hover)

    # Add the plot to the list of all plots
    all_plots.append(p)

# Create a grid of all the plots and show it
grid = gridplot(all_plots, ncols=2)
show(grid)
