import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from analyze import analyze_dataset #Import analyze.py . check the analyze.py
from prepro import object_to_numeric 
from prepro import null_values 

dataset = pd.read_csv("bigmart.csv") #ّّImport dataset bigmart_sale 

analyze_dataset(dataset)  #It does as the name suggests 

dataset = object_to_numeric(dataset) #It does as the name suggests 
dataset = null_values(dataset,-1)
analyze_dataset(dataset)
