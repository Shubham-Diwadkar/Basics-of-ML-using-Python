# Program for implementing Support Vector Machine on 2 columns 

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Giving the file path to access the cancer.csv file
file_path = r"YOUR_FILE_PATH\cancer.csv"

# Reading the .csv file
x = pd.read_csv(file_path)

# Converting DataFrame(x) into numpy array
a = np.array(x)

# Accessing all the data of rows starting starting from 30th column 
y = a[:,30]

# Stacking all the values of State column and Total_Rate column in a numpy array
x=np.column_stack((x.State,x.Total_Rate))

# Defining the shape of all stacked data values
x.shape

# Prints all the values of stacked data values and values of y
print (x),(y)
