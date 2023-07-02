import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("bigmart.csv") #import dataset bigmart_sale 

print("------Data Shape-----\n\n",dataset.shape)
print("\n\n-----Data head----- \n\n",dataset.head())
print("\n\n\n-----data information-----\n",dataset.info)
print("\n\n\n Data size :>>>>> ",dataset.size)

