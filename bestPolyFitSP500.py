import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

# Define the symbols for the S&P500 companies
symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'META']

# Download the historical stock prices for each symbol
data = yf.download(symbols, start='2019-01-01', end='2023-02-28')['Adj Close']

# Choose a stock to analyze
stock_data = data['AAPL']

# Get the closing price data
prices = stock_data.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(np.arange(len(prices)), prices, test_size=0.2, random_state=42)

# Define the maximum degree of the polynomial function to test
max_degree = 10

# Initialize lists to store the model performance
train_mse = []
test_mse = []
aic = []

# Loop over different polynomial degrees and fit the models
for degree in range(1, max_degree+1):
    # Perform polynomial curve fitting of the specified degree
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train.reshape(-1, 1))
    X_poly_test = poly.fit_transform(X_test.reshape(-1, 1))
    model = sm.OLS(y_train, X_poly_train).fit()
    
    # Evaluate the model on the training and testing sets
    y_pred_train = model.predict(X_poly_train)
    y_pred_test = model.predict(X_poly_test)
    train_mse.append(mean_squared_error(y_train, y_pred_train))
    test_mse.append(mean_squared_error(y_test, y_pred_test))
    
    # Calculate the AIC for the model
    aic.append(model.aic)
    
# Find the best model based on AIC
best_degree = np.argmin(aic) + 1
print(f"Best degree based on AIC: {best_degree}")

# Perform polynomial curve fitting of the best model
poly = PolynomialFeatures(degree=best_degree)
X_poly = poly.fit_transform(np.arange(len(prices)).reshape(-1, 1))
model = sm.OLS(prices, X_poly).fit()

# Create a polynomial function from the model coefficients
p = np.poly1d(model.params)

# Evaluate the polynomial function at some points
x_new = np.linspace(0, len(prices), 100)
y_new = p(np.arange(len(prices)))
y_new_interp = p(x_new)

# Plot the data and the fitted curve
plt.plot(np.arange(len(prices)), prices, label='Data')
plt.plot(x_new, y_new_interp, label='Fitted Curve')
plt.legend()
plt.show()

# Plot the mean squared error for different polynomial degrees
plt.plot(range(1, max_degree+1), train_mse, label='Train MSE')
plt.plot(range(1, max_degree+1), test_mse, label='Test MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Plot the AIC for different polynomial degrees
plt.plot(range(1, max_degree+1), aic, label='AIC')
plt.xlabel('Polynomial Degree')
plt.ylabel('AIC')
plt.legend()
plt.show()

"""
The AIC (Akaike Information Criterion) is a statistical measure of the relative quality of a model for a given set of data. 
It provides a way to compare different models and choose the one that best balances complexity and goodness of fit. 
A lower AIC value indicates a better fit, and the model with the lowest AIC value is usually chosen as the best fit for the data.
"""

"""
First, we import the necessary libraries: yfinance for downloading the stock data, numpy for mathematical operations, 
matplotlib for plotting graphs, and statsmodels for calculating the AIC value.

Next, we define the stock symbols for the S&P500 companies we want to analyze.

We use yfinance to download the historical stock prices for each symbol between the start and end dates specified.

We choose one of the stocks to analyze by selecting the 'Adj Close' data for that stock.

We define a range of x-values corresponding to the number of trading days for the selected stock.

We define a function polyfit_and_plot that takes the closing price data for the selected stock, the range of x-values, 
and the degree of the polynomial function to fit the data. The function then performs polynomial curve fitting of the 
pecified degree, creates a polynomial function from the coefficients, evaluates the function at some points, and plots the 
data and the fitted curve. Finally, the function returns the AIC value of the fitted curve.

We call the polyfit_and_plot function for each stock in the S&P500, and print the AIC value for each fitted curve.

Finally, we print the stock with the lowest AIC value, which indicates the best fit.
"""