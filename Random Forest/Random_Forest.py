import numpy as np
import pandas as pd

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, :5].values
X = dataset.iloc[:, -1].values
