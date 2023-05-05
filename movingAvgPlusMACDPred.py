import os
import datetime
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

# Get today's date
today = datetime.datetime.now().date()

# Subtract 365 days from today's date
one_year_ago = today - datetime.timedelta(days=365)

# Use the date one year ago as the start parameter in yf.download()
data = yf.download(stock_ticker, start=one_year_ago)

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

    # Calculate the MACD line, signal line, and histogram
    last_year.loc[:,'MACD_Line'] = last_year['Close'].ewm(span=12).mean() - last_year['Close'].ewm(span=26).mean()
    last_year.loc[:,'Signal_Line'] = last_year['MACD_Line'].ewm(span=9).mean()
    last_year.loc[:,'Histogram'] = last_year['MACD_Line'] - last_year['Signal_Line']

    # Split the data into X (features) and y (target)
    X = last_year[['MACD_Line', 'Signal_Line', 'Histogram']]
    y = last_year['Close']

    # Create an HistGradientBoostingRegressor instance
    model = HistGradientBoostingRegressor()

    # Fit the model with the data
    model.fit(X, y)

    # Make predictions for the next 30 days
    future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
    future_data = pd.DataFrame(index=future_dates, columns=['MACD_Line','Signal_Line','Histogram'])
    future_data['MACD_Line'] = last_year['MACD_Line'].iloc[-1]
    future_data['Signal_Line'] = last_year['Signal_Line'].iloc[-1]
    future_data['Histogram'] = last_year['Histogram'].iloc[-1]

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

    # Recalculate MACD line, signal line, and histogram for the next 30 days
    predictions_df.loc[:,'MACD_Line'] = predictions_df['Close'].ewm(span=12).mean() - predictions_df['Close'].ewm(span=26).mean()
    predictions_df.loc[:,'Signal_Line'] = predictions_df['MACD_Line'].ewm(span=9).mean()
    predictions_df.loc[:,'Histogram'] = predictions_df['MACD_Line'] - predictions_df['Signal_Line']

    # Create a new column in the predictions_df DataFrame to store the buy/sell signals, with a default value of "hold"
    predictions_df['Signal'] = 'hold'

    # Iterate through the predictions_df DataFrame and check the values of the MACD_Line and Signal_Line columns
    for i, row in predictions_df.iterrows():
        if i == 0:
            continue
        if row['MACD_Line'] > row['Signal_Line']:
            predictions_df.at[i, 'Direction'] = 'up'
        elif row['MACD_Line'] < row['Signal_Line']:
            predictions_df.at[i, 'Direction'] = 'down'

    # Set the style to dark theme
    style.use('dark_background')

    # Create the plot
    fig, axs = plt.subplots(2, 1,)

    # Plot the predicted close prices for the next 30 days
    axs[0].plot(predictions_df.index, predictions_df['Close'], color='green' if predictions_df['Close'][-1] >= last_year['Close'][-1] else 'red', label='Predicted')
    axs[0].plot(last_year.index, last_year['Close'], color='blue', label='Actual')
    axs[0].set_title(stock_ticker + " MACD Price Prediction")
    axs[0].set_xticks([])
    axs[0].legend(loc='upper right')

    # Plot the MACD line, signal line, and histogram
    axs[1].plot(predictions_df.index, predictions_df['MACD_Line'], label='MACD Line', color='tab:green')
    axs[1].plot(predictions_df.index, predictions_df['Signal_Line'], label='Signal Line', color='tab:red')
    axs[1].bar(predictions_df.index, predictions_df['Histogram'], label='Histogram', color='tab:blue')
    axs[1].set_title('')
    axs[1].legend(loc='lower left')

    # Create buy and sell signals
    signals = predictions_df[predictions_df['Direction'] != predictions_df['Direction'].shift()].copy()

    # Plot the signal values as scatter points on the second subplot, using a different color for buys and sells
    ax2 = plt.subplot(2, 1, 2)
    ax2.scatter(signals[signals['Direction'] == 'up'].index, signals[signals['Direction'] == 'up']['Signal_Line'], color='green', label='buy')
    ax2.scatter(signals[signals['Direction'] == 'down'].index, signals[signals['Direction'] == 'down']['Signal_Line'], color='red', label='sell')
    
    # Set the x-axis to show dates
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # Set the x-axis limits to be the same for both subplots
    axs[0].set_xlim(predictions_df.index[0], predictions_df.index[-1])
    axs[1].set_xlim(predictions_df.index[0], predictions_df.index[-1])

    #Set the y-axis to show labels
    axs[0].set_ylabel('Price (USD)')
    axs[1].set_ylabel('Moving Average Conver/Diver')

    # Show the plot
    plt.show()