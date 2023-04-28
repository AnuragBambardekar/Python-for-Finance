import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf

# Define time frame
start = dt.datetime(2020,1,1)
end = dt.datetime.now()

# Get user input for the moving average window
ma_window = int(input("Enter the moving average window value: "))

# Define the Ticker symbol
symbol = "AMZN"

# Load the data for the selected Ticker symbol and time frame
df = yf.Ticker(symbol).history(start=start, end=end)

# Calculate the moving average based on user input
ma = df['Close'].rolling(window=ma_window).mean()

# Plot the candlestick chart with the moving average
mpf.plot(df, type="candle", mav=ma_window)

# Add the selected moving average to the chart
# plt.plot(ma, label=f"{ma_window} Day Moving Average")
# plt.legend(loc="upper left")
# plt.show()
