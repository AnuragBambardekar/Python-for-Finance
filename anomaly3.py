import yfinance as yf
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import *
import plotly.graph_objects as go

data = yf.download("TSLA")['Close']

data = validate_series(data)
volatility_detector = VolatilityShiftAD(c=6.0, side="positive", window=30)
anomalies = volatility_detector.fit_detect(data)

# Create a Plotly figure
fig = go.Figure()

# Plot the data as a line
fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines', name='TSLA Close Price'))

# Add red markers at the anomaly points
anomaly_dates = [date for date, anomaly in anomalies.items() if anomaly]
anomaly_values = [data[date] for date in anomaly_dates]
fig.add_trace(go.Scatter(x=anomaly_dates, y=anomaly_values, mode='markers', marker=dict(color='red'), name='Anomalies'))

# Set the axis labels and title
fig.update_layout(title='TSLA Close Price', xaxis_title='Date', yaxis_title='Price')

# Show the Plotly figure
fig.show()
