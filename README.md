# Basics-of-ML-using-Python

1. Install Required Libraries: Before getting started with ML in Python, you'll need to install some essential libraries such as NumPy, Pandas, and Scikit-learn. You can install them using package managers like pip or conda.
  
2. Load Data: ML algorithms require data for training and testing. You can load data from various sources such as CSV files, databases, or APIs. Python provides libraries like Pandas to read and manipulate data.

3. Preprocess the Data: Data preprocessing is an important step in ML. It involves cleaning, transforming, and normalizing the data to make it suitable for training ML models. Common preprocessing steps include handling missing values, scaling features, and encoding categorical variables.

4. Split Data into Training and Testing Sets: To evaluate the performance of ML models, it's necessary to split the data into training and testing sets. The training set is used to train the model, while the testing set is used to evaluate its performance on unseen data.

5. Choose an ML Algorithm: There are various ML algorithms available for different types of problems, such as linear regression for regression tasks, decision trees for classification tasks, and clustering algorithms for unsupervised learning. Select the algorithm that is appropriate for your problem.

6. Train the Model: After choosing an algorithm, you need to train the ML model using the training data. In Python, you can use libraries like Scikit-learn to instantiate the model, fit it to the training data, and learn the underlying patterns.

7. Evaluate the Model: Once the model is trained, you can evaluate its performance using the testing data. Common evaluation metrics include accuracy, precision, recall, and F1 score for classification problems, and mean squared error (MSE) or R-squared for regression problems.

8. Tune the Model: ML models often have hyperparameters that control their behavior. You can fine-tune these hyperparameters to improve the model's performance. Techniques like grid search or random search can be used to find the optimal hyperparameters.

9. Make Predictions: After the model is trained and evaluated, you can use it to make predictions on new, unseen data. Provide the new data to the model, and it will generate predictions based on the patterns it learned during training.

## Dataset

The California Housing Dataset is used for training and testing the model.
The target variable is the median house value.

[Note: As the Datasets is too large to be uploaded here I will be providing the link for the same.]

All the Datasets which are been used in this reposetory are downloaded from below link:

Click here to [Download](https://drive.google.com/drive/folders/1Wsq93CEka5mkcGMj9gUbIWUPe1e0lJpw?usp=sharing) the datasets folder

This link contains datasets which are as follows:
1. California Housing Dataset
2. Diabetes Dataset
3. Cancer Dataset

## Prerequisites

To run the code in this repository, you need to have the following dependencies installed:

- Python (version 3.x)
  I have used Python (version 3.8.0) [Click Here](https://www.python.org/downloads/release/python-380/)
- NumPy
  ```
  pip install numpy
  ```
- Pandas
  ```
  pip install pandas
  ```
- scikit-learn
  ```
  pip install scikit-learn
  ```
- matplotlib.pyplot
  ```
  pip install matplotlib
  ```

### Linear Regression Model on California Housing Dataset

This file contains code for training and evaluating a linear regression model on the California Housing Dataset. The model predicts the median house value based on various housing-related features.

#### Output

![image](https://github.com/Shubham-Diwadkar/Basics-of-ML-using-Python/assets/125255910/5191935d-aefe-4be8-9a4f-c6a332d9bf55)

![image](https://github.com/Shubham-Diwadkar/Basics-of-ML-using-Python/assets/125255910/5c14ec55-5802-493d-988d-c39459fbd9a8)

### Logistic Regression Model on Daibetes Dataset

This file contains an implementation of the logistic regression algorithm from scratch and compares its accuracy with the logistic regression model from scikit-learn on a diabetes dataset.

#### Output

![image](https://github.com/Shubham-Diwadkar/Basics-of-ML-using-Python/assets/125255910/c0769932-13b1-42a9-80fa-1229317502bd)

### Support Vector Machine(Part 1 - 1 column) on Cancer Dataset

This program reads a CSV file containing cancer data and extracts a single column.
It then performs some operations on the column data.

The main steps of the program are as follows:
- Load the cancer dataset from a CSV file using Pandas.
- Extract a single column from the dataset.
- Perform operations on the extracted column data.
- In this program, the shape of the column is printed.
- Print the extracted column data.

#### Output

![image](https://github.com/Shubham-Diwadkar/Basics-of-ML-using-Python/assets/125255910/77f876e2-3ecf-4b22-bea0-387aeec9fc8e)

### Support Vector Machine (Part 2 - 2 columns) on Cancer Dataset

This program reads a CSV file containing cancer data and extracts two columns.
It then performs some operations on the column data.

The main steps of the program are as follows:
- Load the cancer dataset from a CSV file using Pandas.
- Extract two columns from the dataset.
- Perform operations on the extracted columns data.
- In this program, the shape of the columns is printed.
- Print the extracted columns data.

#### Output

![image](https://github.com/Shubham-Diwadkar/Basics-of-ML-using-Python/assets/125255910/d0c55863-4c69-447b-87ac-abc0d7dcdcd0)

### Decision Tree on Balance Scale Dataset

This program demonstrates the use of a Decision Tree Classifier to classify the balance-scale dataset.

The `Decision Tree.py` file performs the following steps:
- Imports the necessary libraries and modules: NumPy, Pandas, scikit-learn metrics, scikit-learn model_selection, and scikit-learn DecisionTreeClassifier
- Defines functions to import the dataset, split the dataset into training and testing sets, train the model using the Gini index criterion, train the model using the entropy criterion, make predictions, and calculate accuracy.
- Loads the balance-scale dataset from the UCI Machine Learning Repository.
- Splits the dataset into training and testing sets.
- Trains the decision tree classifier using the Gini index criterion.
- Trains the decision tree classifier using the entropy criterion.
- Prints the results using the Gini index criterion: predicted values, confusion matrix, accuracy, and classification report.
- Prints the results using the entropy criterion: predicted values, confusion matrix, accuracy, and classification report.

#### Output

![image](https://github.com/Shubham-Diwadkar/Basics-of-ML-using-Python/assets/125255910/218b21fe-c9aa-4178-ba56-2b09ceaacdab)
