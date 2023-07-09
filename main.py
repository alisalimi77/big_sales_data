import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from analyze import analyze_dataset #Import analyze.py . check the analyze.py
from prepro import DataPreprocessor


dataset = pd.read_csv("bigmart.csv") #ّّImport dataset for example bigmart_sale "bigmart.csv" 

analyze_dataset(dataset)  #It does as the name suggests 

# preprocess = DataPreprocessor()

# prepro_dataset = preprocess.object_to_numeric(dataset) #It does as the name suggests
# prepro_dataset = preprocess.handle_null_values(prepro_dataset,-1) #Check the null values are not null values 
# prepro_dataset = preprocess.standardize_dataset(prepro_dataset) #Standardization the dataset     
# prepro_dataset = preprocess.normalize_dataset(prepro_dataset) #Normalaization.

# analyze_dataset(prepro_dataset)

