import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Define the symbols for the S&P500 companies
symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB']

# Download the historical stock prices for each symbol
data = yf.download(symbols, start='2010-01-01', end='2022-02-27')['Adj Close']

# Choose a stock to analyze
stock_data = data['AAPL']

# Get the closing price data
prices = stock_data.values

# Perform polynomial curve fitting of degree 3
coefficients = np.polyfit(range(len(prices)), prices, 3)

# Create a polynomial function from the coefficients
p = np.poly1d(coefficients)

# Evaluate the polynomial function at some points
x_new = np.linspace(0, len(prices), 100)
y_new = p(x_new)

# Plot the data and the fitted curve
plt.plot(range(len(prices)), prices, label='Data')
plt.plot(x_new, y_new, label='Fitted Curve')
plt.legend()
plt.show()


""" 
The graph obtained from polynomial curve fitting of S&P500 stock data shows the relationship between time (x-axis) and stock price (y-axis). 
The fitted curve represents a mathematical function that approximates the underlying trend of the data. By examining the graph, you can gain 
insights into the behavior of the stock price over time, such as whether it has increased, decreased, or remained relatively constant.

Additionally, the graph can help you to identify patterns in the data that may not be apparent from a simple table of numbers. For example, 
you might observe that the stock price follows a cyclical pattern, or that it tends to increase rapidly during certain time periods.

Furthermore, the graph can be useful in making predictions about future stock prices. By extrapolating the fitted curve beyond the range of 
the available data, you can estimate what the stock price might be at some future point in time. However, it is important to note that these 
predictions are only as good as the underlying model and assumptions used to generate them, and that stock prices are subject to a variety of 
unpredictable external factors such as economic conditions, news events, and market sentiment.

"""

"""
Polynomial curve fitting and moving averages are both techniques used in financial analysis to identify trends and patterns in stock prices 
over time. However, they differ in their approach and the insights they provide.

Moving averages are a statistical technique that involve calculating the average of a series of prices over a certain time period (e.g. the 
last 50 days). This average value is then plotted on a graph to show how the stock price has changed over time relative to this moving 
average. By comparing the stock price to the moving average, you can identify trends and potential buy/sell signals. For example, if the 
stock price is consistently above the moving average, this may indicate a bullish trend and a good time to buy, whereas if the stock price 
is consistently below the moving average, this may indicate a bearish trend and a good time to sell.

Polynomial curve fitting, on the other hand, involves fitting a mathematical function to the historical stock price data using a polynomial 
equation. This allows you to model the underlying trend of the data more precisely, and can provide insights into the shape and direction of 
the trend over time. Polynomial curve fitting is also more flexible than moving averages, as it allows you to fit curves of varying degrees 
(e.g. linear, quadratic, cubic) to the data to capture more complex trends.

In summary, moving averages provide a simple way to identify trends in stock prices over time, while polynomial curve fitting allows for 
a more detailed analysis of the underlying trend and can capture more complex patterns in the data. Both techniques can be useful in 
inancial analysis, and may be used together to gain a more comprehensive understanding of the behavior of a stock over time.

"""