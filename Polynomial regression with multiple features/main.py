"""
This script aims to combine multiple features and non-linear fits. In this example the data set has a non-linear trend
and there are two features "x1" and "x2" per label "y". Two features were selected purely so that the model could be
visualised in 3d space. With that being said this method can be extended to more than two features.

However, as the number features in our basis are increased (in this case adding a polynomial degree) the number of
parameters (weights) the algorithm has to solve for increases exponentially. This means that this method can often be
limited by the curse of dimensionality where very complex functions are being modelled and the dataset contains high
numbers of features. In cases like these we would have to look towards alternative methods to solve our modelling
problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm

pi = math.pi

# Create a new figure for plotting
fig = plt.figure()
# Add a 3D subplot to the figure
ax = fig.add_subplot(111, projection='3d')

# Generate linearly spaced values for x1 and x2 from -1 to 1 with 100 steps
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
# Create a 2D meshgrid for x1 and x2
[X1, X2] = np.meshgrid(x1, x2)

# Compute the Y values for the grid using a cosine-sine function and add some random noise
Y = np.cos(2 * pi * X1) * np.sin(2 * pi * 1 * X2) + 0.1 * np.random.randn(X1.shape[0], X1.shape[1])

# Plot the surface using the X1, X2, and Y matrices
surf = ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title("original data")
plt.show()

# Extract the number of rows and columns from X1 (or X2 since they are the same shape)
nr, nc = X1.shape

# Define a value M which will likely be used for polynomial fitting
M = 8
# Reshape X1 and X2 to add two new dimensions for polynomial powers
X1t = X1.reshape(nr, nc, 1, 1)
X2t = X2.reshape(nr, nc, 1, 1)
# Create matrices I and J of size M+1 representing the polynomial degrees
I = np.arange(M + 1).reshape(1, 1, -1, 1)
J = np.arange(M + 1).reshape(1, 1, 1, -1)

# Create a 4D matrix X which represents all combinations of polynomial degrees for X1 and X2
X = X1t ** I * X2t ** J
# Reshape X to have shape (nr*nc, (M+1)**2)
X = X.reshape(nr * nc, (M + 1) ** 2)


# Compute the optimal coefficients theta using linear regression
theta = np.linalg.solve(X.T @ X, X.T @ Y.reshape(-1, 1))

# Compute the predicted Y values using the found coefficients
yhat = X @ theta

# Reshape yhat back to the shape of Y for plotting
Yhat = yhat.reshape(Y.shape)

# Create a new figure for plotting the predicted values
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the predicted surface using the X1, X2, and Yhat matrices
surf = ax.plot_surface(X1, X2, Yhat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.title("model")
plt.show()