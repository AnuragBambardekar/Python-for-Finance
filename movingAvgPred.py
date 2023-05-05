import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.dates as mdates
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting

# Ask the user for the stock ticker symbol
stock_ticker = input("Enter the stock ticker symbol: ")

# Download the stock data for the last year
data = yf.download(stock_ticker, start='2022-01-23')

if data.empty:
    print("No data available for the stock ticker symbol: ", stock_ticker)
else:
    # Convert the date column to a datetime object
    data['Date'] = pd.to_datetime(data.index)

    # Set the date column as the index
    data.set_index('Date', inplace=True)

    # Sort the data by date
    data.sort_index(inplace=True)

    # Get the data for the last year
    last_year = data.iloc[-365:].copy()

    # Calculate the 200-day moving average
    last_year.loc[:,'200MA'] = last_year['Close'].rolling(window=200).mean()

    # Split the data into X (features) and y (target)
    X = last_year[['200MA']]
    y = last_year['Close']

    # Create an HistGradientBoostingRegressor instance
    model = HistGradientBoostingRegressor()

    # Fit the model with the data
    model.fit(X, y)

    # Make predictions for the next 30 days
    future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
    future_data = pd.DataFrame(index=future_dates, columns=['200MA'])
    future_data['200MA'] = last_year['200MA'].iloc[-1]
    predictions = model.predict(future_data)
    predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])

    # Calculate the standard deviation of the last year's close prices
    std_dev = last_year['Close'].std()

    # Generate random values with a standard deviation of 0.5 * the last year's close prices standard deviation
    random_values = np.random.normal(0, 0.2 * std_dev, predictions.shape)

    # Add the random values to the predicted prices
    predictions += random_values
    predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])

    # Concatenate the last_year and predictions dataframes
    predictions_df = pd.concat([last_year, predictions_df])

    # Calculate 200 day moving average
    predictions_df.loc[:,'MA_200'] = predictions_df['Close'].rolling(window=200).mean()

    # Set the style to dark theme
    style.use('dark_background')

    # Create the plot
    fig, ax = plt.subplots()

    # Plot the predicted close prices for the next 30 days
    ax.plot(predictions_df.index, predictions_df['Close'], color='green' if predictions_df['Close'][-1] >= last_year['Close'][-1] else 'red', label='Predicted')

    # Plot the actual close prices for the last year
    ax.plot(last_year.index, last_year['Close'], color='b', label='Actual')

    ax.plot(predictions_df.index, predictions_df['MA_200'], color='white', label='200 Day MA')

    # Set x-axis as date format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%B %D %Y"))
    plt.xticks(rotation=45)

    # Set the x-axis label
    plt.xlabel('Date')

    # Set the y-axis label
    plt.ylabel('Price (USD)')

    # Set the plot title
    plt.title(stock_ticker + ' Moving Average Price Prediction')

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()