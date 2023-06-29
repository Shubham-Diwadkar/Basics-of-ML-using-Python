# Linear Regression model on California Housing Dataset

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California housing dataset
california_housing = fetch_california_housing()

# Extract the features (X) and target variable (y)
X = california_housing.data
y = california_housing.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)

# Create a linear regression model
reg = linear_model.LinearRegression()

# Train the model using the training data
reg.fit(X_train, y_train)

# Print the coefficients of the model
print("Coefficients: ", reg.coef_)

# Calculate and print the variance score (R^2) of the model on the test set
print("Variance score:{}".format(reg.score(X_test, y_test)))

plt.style.use("fivethirtyeight")

# Create a scatter plot of the residual errors
plt.scatter(reg.predict(X_test), reg.predict(X_test)- y_test, color = "blue", s = 10, label = "Test Data")

# Add a horizontal line at y=0 for reference
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

# Add a legend to the plot
plt.legend(loc = "upper right")

# Set the title of the plot
plt.title("Residual errors")

# Show the plot
plt.show()
