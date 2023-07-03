import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def importdata():
    balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-'+'databases/balance-scale/balance-scale.data', sep= ',', header = None)

    print("\nDataset Length: ", len(balance_data))

    print("\nDataset Shape: ", balance_data.shape)

    print("\nDataset: ", balance_data.head())

    return balance_data

def splitdataset(balance_data):
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

    return X, Y, X_train, X_test, y_train, y_test

def train_using_gini(X_train, X_test, y_train):
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 3, min_samples_leaf = 5)

    clf_gini.fit(X_train, y_train)

    return clf_gini

def train_using_entropy(X_train, X_test, y_train):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)

    clf_entropy.fit(X_train, y_train)

    return clf_entropy

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)

    print("\nPredicted values: ")
    print("\n", y_pred)
    return y_pred

def cal_accuracy(y_test, y_pred):
    print("\nConfusion Matrix: ", confusion_matrix(y_test, y_pred))

    print("\nAccuracy: ", accuracy_score(y_test, y_pred) * 100)

    print("\nRepport: ", classification_report(y_test, y_pred))

def main():
    data = importdata()

    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    clf_gini = train_using_gini(X_train, X_test, y_train)

    clf_entropy = train_using_entropy(X_train, X_test, y_train)

    print("\nResults using Gini Index: ")

    y_pred_gini = prediction(X_test, clf_gini)

    cal_accuracy(y_test, y_pred_gini)

    print("\nResult using Entropy: ")

    y_pred_entropy = prediction(X_test, clf_entropy)

    cal_accuracy(y_test, y_pred_entropy)

if __name__ == "__main__":
    main()