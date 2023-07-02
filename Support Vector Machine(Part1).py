# Program for 1 column

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = r"D:\Python Programing\All Datasets\cancer.csv"
x = pd.read_csv(file_path)
a = np.array(x)
y = a[:, 30]

x = np.column_stack((x.State))

x.shape

print(x),(y)