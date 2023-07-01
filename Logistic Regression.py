# Simple Logistic Regression Model on daibetes Dataset

# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

# Logistic Regression Class
class LogisticRegression():

    # Constructor of the class
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Custom Function for training the model
    def fit(self, X, Y):
        
        # Calculating the number of rows(traning examples) and number of columns(training features)
        self.m, self.n = X.shape

        # Initializes weight vector of 1D array of zeros with size "number of columns"
        self.W = np.zeros(self.n)

        # Initializing bias term to 0
        self.b = 0

        # Assigning the input data X and Y to the objects respectively
        self.X = X
        self.Y = Y

        # For each iteration model's weights and bais are upadated
        for i in range(self.iterations):

            # Calling the update_weights() function
            self.update_weights()

        # returns the updated Logistic Regression object
        return self

    # Custom Method for updating the weights and bias on the current iteration
    def update_weights(self):

        # Calculating the predicted probabilities of the positive class
        A = 1 / (1 + np.exp(-(self.X.dot(self.W) + self.b)))

        # Calculating the difference between predicted probabilities(A) and the true labels(self.Y).
        # T is used to transpose the true labels
        tmp = (A - self.Y.T)

        # Matching the dimensions of the feature matrix, reshapes the column vector `tmp` --> 1D array with length = number of training examples `self.m`.
        tmp = np.reshape(tmp, self.m)

        # Calculating the gradient of the weights by taking the dot product
        dW = np.dot(self.X.T, tmp)

        # Calculating the gradient of the bias term
        db = np.sum(tmp) / self.m

        # Updating the weight vector `self.W` and the bias term `self.b`
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self

    # Custom Method for predicting the ouputs of trained model
    def predict(self, X):

        # Calculating the output of the logistic function for each example
        Z = 1/(1 + np.exp(-(X.dot(self.W) + self.b)))

        # Converting the predicted probabilities Z into binary predictions.
        # Assigns 1 to those probabilities > 0.5, indicating the positive class
        # Assigns 0 to the probabilities <= 0.5, indicating the negative class.
        Y = np.where(Z > 0.5, 1, 0)

        # Returning binary predictions for the input data
        return Y

# Code's entry point
def main():

    # The path to the CSV file containing the dataset
    file_path = r"YOUR_PATH_TO_THE_FILENAME.csv"

    # Reads the CSV file into a pandas DataFrame.
    df = pd.read_csv(file_path)

    # Extracting the feature matrix X and the target vector Y from the DataFrame.
    # The iloc function is used to select the columns based on their indices.
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1:].values

    # Spliting the data into training and testing sets using the train_test_split function from scikit-learn.
    # Assigning 2/3 of the data to the training set, 1/3 of the data to the testing set.
    # The random_state argument is set to 0 to ensure reproducibility.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

    # Creating an instance of the LogisticRegression class with a learning rate of 0.01 and 1000 iterations.
    model = LogisticRegression(learning_rate=0.01, iterations=1000)

    # Fitting the logistic regression model to the training data. 
    # Learning the weights and bias using the training data.
    model.fit(X_train, Y_train)

    # Creating an another instance of the LogisticRegression class with a learning rate of 0.01 and 1000 iterations.
    model1 = LogisticRegression(learning_rate=0.01, iterations=1000)

    # Fitting the second logistic regression model to the training data.
    model1.fit(X_train, Y_train)

    # Making predictions on the test data by calling the predict method
    Y_pred = model.predict(X_test)
    Y_pred1 = model1.predict(X_test)

    # Initializing counters for correctly classified examples
    correctly_classified = 0
    correctly_classified1 = 0
    count = 0

    # Iterating over the predicted classes
    for count in range(np.size(Y_pred)):

        # If true
        if Y_test[count] == Y_pred[count]:
            # Incrementing the counter for first instance 
            correctly_classified = correctly_classified + 1

        # If true
        if Y_test[count] == Y_pred1[count]:
            # Incrementing the counter for second instance
            correctly_classified1 = correctly_classified1 + 1

        # Third counter is incremented everytime 
        count = count + 1

    # Prints the accuracy of the first model
    print("\nAccuracy on test set by our model: ", (correctly_classified / count) * 100)

    # Prints the accuracy of the second model
    print("\nAccuracy on test set by sklearn model: ", (correctly_classified1 / count) * 100)

# For running a stand alone program
if __name__ == "__main__":
    
    # Calling the main method/function
    main()
