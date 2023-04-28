import yfinance as yf
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
import pandas as pd

# retrieve stock data
data = yf.download("AAPL", start="2022-01-01", end="2023-04-28")

# ask user for window values
windows = input("Enter window values for moving averages (comma separated): ").split(",")
windows = [int(window.strip()) for window in windows]

# create Bokeh figure
p = figure(x_axis_type="datetime", title="AAPL Candlestick Chart")

# calculate candlestick properties
inc = data.Close > data.Open
dec = data.Open > data.Close
w = 12*60*60*1000 # width of each candlestick (12 hours in milliseconds)
midpoint = (data.Open + data.Close) / 2
height = abs(data.Close - data.Open)

# add candlestick glyphs
p.segment(data.index, data.High, data.index, data.Low, color="black")
p.vbar(data.index[inc], w, data.Open[inc], data.Close[inc], fill_color="#00FF00", line_color="black")
p.vbar(data.index[dec], w, data.Open[dec], data.Close[dec], fill_color="#FF0000", line_color="black")

# add moving average lines
colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
for i, window in enumerate(windows):
    # calculate moving average
    ma = data.Close.rolling(window=window).mean()

    # add line glyph
    p.line(data.index, ma, color=colors[i], legend_label=f"MA{window}")

# configure legend
p.legend.location = "top_left"
p.legend.click_policy = "hide"

# display plot
show(p)
