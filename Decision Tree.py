# Program on simple decision tree on a dataset from UC Irvine Machine Learning Repository.

# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Custom Method for importing the dataset
def importdata():

    # Reading the dataset from a CSV file located at the given URL.
    balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-'+'databases/balance-scale/balance-scale.data', sep= ',', header = None)

    # Printing the length of the dataset, which corresponds to the number of rows in the DataFrame
    print("\nDataset Length: ", len(balance_data))

    # Printing the shape of the dataset, which represents the number of rows and columns in the DataFrame
    print("\nDataset Shape: ", balance_data.shape)

    # Printing the first few rows of the dataset
    print("\nDataset: ", balance_data.head())

    return balance_data

# Custom Method for splitting the dataset into training and testing datasets
def splitdataset(balance_data):

    # Extracting the feature columns from the balance_data DataFrame
    X = balance_data.values[:, 1:5]

    # Extracting the target column from the balance_data DataFrame
    Y = balance_data.values[:, 0]

    # Splitting the features (X) and target (Y) arrays into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

    return X, Y, X_train, X_test, y_train, y_test

# Custom Method for training a decision tree classifier using the Gini impurity criterion
def train_using_gini(X_train, X_test, y_train):

    # Createing an instance of the DecisionTreeClassifier class from scikit-learn
    # The criterion parameter is set to "gini" to indicate that the Gini impurity criterion should be used
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 3, min_samples_leaf = 5)

    # Training the decision tree classifier using the training data
    clf_gini.fit(X_train, y_train)

    return clf_gini

# Custom Method for training a decision tree classifier using the entropy criterion
def train_using_entropy(X_train, X_test, y_train):

    # Creating an instance of the DecisionTreeClassifier class from scikit-learn.
    # The criterion parameter is set to "entropy" to indicate that the entropy criterion should be used
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)

    # Training the decision tree classifier using the training data
    clf_entropy.fit(X_train, y_train)

    return clf_entropy

# Custom Method for performing predictions on test data
def prediction(X_test, clf_object):

    # Using the trained classifier clf_object to predict the target values for the test data X_test
    y_pred = clf_object.predict(X_test)

    # Printing a message indicating that the following lines will display the predicted values.
    print("\nPredicted values: ")

    # Printing the predicted values on a new line, the y_pred array contains the predicted values for the test data.
    print("\n", y_pred)
    
    return y_pred

# Custom Method for calculating and printing various metrics to evaluate the accuracy of the predicted values
def cal_accuracy(y_test, y_pred):

    # Calculating the confusion matrix using the true target values y_test and the predicted values y_pred
    print("\nConfusion Matrix: ", confusion_matrix(y_test, y_pred))

    # Calculating the accuracy of the predictions by comparing the true target values y_test with the predicted values y_pred
    print("\nAccuracy: ", accuracy_score(y_test, y_pred) * 100)

    # Generating a classification report that includes precision, recall, F1-score, and support for each class
    print("\nRepport: ", classification_report(y_test, y_pred))

# It is a main entry point of the program
def main():

    # Calling  the importdata() function to load the dataset
    data = importdata()

    # Calling the splitdataset() function to split the dataset into training and testing sets
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    # Training a decision tree classifier using the Gini impurity criterion by calling the train_using_gini() function
    clf_gini = train_using_gini(X_train, X_test, y_train)

    # Training a decision tree classifier using the entropy criterion by calling the train_using_entropy() function
    clf_entropy = train_using_entropy(X_train, X_test, y_train)

    # Printing a header indicating the results obtained using the Gini Index criterion
    print("\nResults using Gini Index: ")

    # Calling the prediction() function to make predictions on the test data using the clf_gini model
    y_pred_gini = prediction(X_test, clf_gini)

    # Calling the cal_accuracy() function to evaluate the accuracy of the predictions made by the clf_gini model
    cal_accuracy(y_test, y_pred_gini)

    # Printing a header indicating the results obtained using the entropy criterion
    print("\nResult using Entropy: ")

    # Calling the prediction() function to make predictions on the test data using the clf_entropy model
    y_pred_entropy = prediction(X_test, clf_entropy)

    # Calling the cal_accuracy() function to evaluate the accuracy of the predictions made by the clf_entropy model
    cal_accuracy(y_test, y_pred_entropy)

# For running a stand alone program
if __name__ == "__main__":
    
    # Calling the main method/function
    main()
