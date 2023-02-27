# Polynomial curve fitting is a technique that can be used to model the relationship between two variables.
import numpy as np

# Generate some sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Perform polynomial regression of degree 2
coefficients = np.polyfit(x, y, 2)

# Create a polynomial function from the coefficients
p = np.poly1d(coefficients)

# Evaluate the polynomial function at some points
x_new = np.array([6, 7, 8])
y_new = p(x_new)

print(y_new)
