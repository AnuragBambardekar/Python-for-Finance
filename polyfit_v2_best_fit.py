import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Download AAPL stock data
aapl = yf.Ticker("AAPL")
# df = aapl.history(period="max")

# Download the historical stock prices for each symbol
df = yf.download("AAPL", start='2019-01-01', end='2023-04-27')

# Define x and y variables
x = np.arange(len(df))
y = df['Close']

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Fit polynomial models with degrees from 1 to 10 and evaluate on the validation set
degrees = range(1, 11)
mse_val = []
for deg in degrees:
    poly_fit = np.polyfit(x_train, y_train, deg)
    poly = np.poly1d(poly_fit)
    y_val_pred = poly(x_val)
    mse = mean_squared_error(y_val, y_val_pred)
    mse_val.append(mse)

# Select the best model based on the validation set performance
best_deg = degrees[np.argmin(mse_val)]
poly_fit = np.polyfit(x, y, best_deg)
poly = np.poly1d(poly_fit)

# Plot the curve and the original data
plt.figure(figsize=(10, 6))
plt.plot(df.index, y, label='AAPL')
plt.plot(df.index, poly(x), label=f'Polynomial Fit (deg={best_deg})')
plt.xlabel('Date')
plt.ylabel('Price')

# Predict future prices using the polynomial fit
future_x = np.arange(len(df), len(df)+30)
future_y = poly(future_x)
plt.plot(df.index[-1]+pd.to_timedelta(np.arange(1, 31), 'D'), future_y, 'r-', label='Future Prices')

plt.legend()
plt.show()
