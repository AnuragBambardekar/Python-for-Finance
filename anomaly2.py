import yfinance as yf
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import *
from bokeh.plotting import figure, show
from bokeh.models import DatetimeTickFormatter

data = yf.download("TSLA")['Close']

data = validate_series(data)
volatility_detector = VolatilityShiftAD(c=6.0, side="positive", window=30)
anomalies = volatility_detector.fit_detect(data)

# Create a Bokeh figure
p = figure(title="TSLA Close Price", x_axis_label='Date', y_axis_label='Price')

# Format the x-axis tick labels
p.xaxis.formatter=DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])

# Plot the data as a line
p.line(data.index, data.values, line_width=2)

# Add red circles at the anomaly points
anomaly_dates = [date for date, anomaly in anomalies.items() if anomaly]
anomaly_values = [data[date] for date in anomaly_dates]
p.circle(anomaly_dates, anomaly_values, size=8, color='red')

# Show the Bokeh figure
show(p)
