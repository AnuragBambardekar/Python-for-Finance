import yfinance as yf
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import *
import matplotlib.pyplot as plt

"""
VolatilityShiftAD detects shift of volatility level by tracking the difference between standard deviations at two sliding time windows next to each other. Internally, it is implemented as a pipenet with transformer DoubleRollingAggregate.
"""

data = yf.download("TSLA")['Close']

data = validate_series(data)
volatility_detector = VolatilityShiftAD(c=6.0, side="positive", window=30)
anomalies = volatility_detector.fit_detect(data)

plot(data, anomaly=anomalies, anomaly_color="red")
plt.show()