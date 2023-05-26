import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf
import pandas as pd

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
ticker_list1 = df['Symbol'].to_list()
ticker_list1[0:]

# Call betas
betas = betas('^GSPC', ticker_list1, '2010-01-01', '2023-05-01')

#assigning Beta column to X
X = betas[['Beta']]

#testing number of cluster from 2 to 10 and collecting the silhouette scores
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
scores = []
for n_clusters in range_n_clusters:
    agglom = AgglomerativeClustering(n_clusters=n_clusters)
    agglom.fit(X)
    labels = agglom.labels_
    scores.append(silhouette_score(X, labels))

# average = sum(scores)/len(scores)

# Testing Plots to see optimal number of clusters
# for n_clusters in range_n_clusters:
#     model = AgglomerativeClustering(n_clusters=n_clusters)
#     labels = model.fit_predict(X)
#     # Create scatter plot of data points colored by cluster label
#     plt.scatter(X, betas['Stock_Name'], c=labels, cmap='rainbow')
#     plt.xlabel('Beta')
#     plt.ylabel('Beta')
#     plt.title(f"n_clusters={n_clusters}")
#     cluster_counts = np.bincount(labels)
#     for i in range(n_clusters):
#         print(f"Cluster {i+1} has {cluster_counts[i]} observations")
#     plt.yticks([])
#     plt.show()

# We choose 4
# optimal_n_clusters = 4
# agglom = AgglomerativeClustering(n_clusters=optimal_n_clusters)
# cluster_labels = agglom.fit_predict(X)
# betas['Cluster'] = cluster_labels

# cluster4 = sns.lmplot(data=betas, x='Cluster', y='Beta', hue='Cluster', 
#                 legend=True, legend_out=True)

# sns.violinplot(x='Cluster', y='Beta', data=betas)


optimal_n_clusters = 4
agglom = AgglomerativeClustering(n_clusters=optimal_n_clusters)
cluster_labels = agglom.fit_predict(X)
betas['Cluster'] = cluster_labels

import mpld3
from matplotlib import pyplot as plt

cluster4 = sns.lmplot(data=betas, x='Cluster', y='Beta', hue='Cluster', legend=True, legend_out=True)

# Convert the plot to HTML using mpld3
html = mpld3.fig_to_html(cluster4.fig)

# Save the HTML to a file
with open('myplot.html', 'w') as f:
    f.write(html)

