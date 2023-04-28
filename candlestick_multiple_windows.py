import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf

# Define time frame
start = dt.datetime(2020,1,1)
end = dt.datetime.now()

# Get user input for the moving average windows
ma_windows = input("Enter the moving average window values separated by comma: ").split(",")
ma_windows = [int(i) for i in ma_windows]

# Define the Ticker symbol
symbol = "AMZN"

# Load the data for the selected Ticker symbol and time frame
df = yf.Ticker(symbol).history(start=start, end=end)

# Calculate the moving averages based on user input
ma = df['Close'].rolling(window=max(ma_windows)).mean()

# Plot the candlestick chart with the moving averages
mc = mpf.make_marketcolors(up='g', down='r')
s = mpf.make_mpf_style(marketcolors=mc)
mpf.plot(df, type="candle", mav=ma_windows, style=s)

# Add the selected moving averages to the chart
for ma_window in ma_windows:
    ma = df['Close'].rolling(window=ma_window).mean()
    plt.plot(ma, label=f"{ma_window} Day Moving Average")
plt.legend(loc="upper left")
plt.show()
