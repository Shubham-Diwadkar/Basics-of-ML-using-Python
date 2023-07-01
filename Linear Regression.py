# Linear Regression model on California Housing Dataset

# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Loading the California housing dataset
california_housing = fetch_california_housing()

# Extracting the features (X) and target variable (y)
X = california_housing.data
y = california_housing.target

# Spliting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)

# Creating a linear regression model
reg = linear_model.LinearRegression()

# Training the model using the training data
reg.fit(X_train, y_train)

# Printing the coefficients of the model
print("Coefficients: ", reg.coef_)

# Calculating and printing the variance score (R^2) of the model on the test set
print("Variance score:{}".format(reg.score(X_test, y_test)))

# Setting the plot style to "fivethirtyeight" in Matplotlib
plt.style.use("fivethirtyeight")

# Creating a scatter plot of the residual errors
plt.scatter(reg.predict(X_test), reg.predict(X_test)- y_test, color = "blue", s = 10, label = "Test Data")

# Adding a horizontal line at y=0 for reference
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

# Adding a legend to the plot
plt.legend(loc = "upper right")

# Setting the title of the plot
plt.title("Residual errors")

# Showing the plot
plt.show()
