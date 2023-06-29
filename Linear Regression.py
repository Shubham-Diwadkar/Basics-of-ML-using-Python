import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model, metrics

from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing()

X = california_housing.data
y = california_housing.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)

reg = linear_model.LinearRegression()

reg.fit(X_train, y_train)

print("Coefficients: ", reg.coef_)

print("Variance score:{}".format(reg.score(X_test, y_test)))

plt.style.use("fivethirtyeight")

plt.scatter(reg.predict(X_test), reg.predict(X_test)- y_test, color = "blue", s = 10, label = "Test Data")

plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

plt.legend(loc = "upper right")

plt.title("Residual errors")

plt.show()