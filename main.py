import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


dataset = pd.read_csv("bigmart.csv")

print(dataset.head())
dataset.tail()
print(dataset.describe())


