#Visualizing Stock Data with candlestick charts
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf

#Define time frame
start = dt.datetime(2023,1,1) #January 1, 2023, have to be mindful of whether the company existed on start date or not
end = dt.datetime.now()

df = yf.Ticker("AMZN").history(start=start,end=end) #AAPL
print(df)

# print(df["Open"])
# print(df["Open"].rolling(window=10).mean())

df["15ma"] = (df["Open"].rolling(window=15).mean() )/1.0
df["ma"] = (df["Open"].rolling(window=15).mean() )*1.15
apds = [mpf.make_addplot(df[["15ma","ma"]])]
# print(df)
# print(apds)

# Can set range to plot
# df = df.loc["2023-01-17":]

#Types of plots
mpf.plot(df, type="candle")
mpf.plot(df, type="line")
mpf.plot(df, type="renko")
mpf.plot(df, type="candle", volume=True)
mpf.plot(df, type="candle", volume=True, mav=10)
mpf.plot(df, type="candle", volume=True, mav=(10,3,20))
mpf.plot(df, type="candle", addplot=apds)

