import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from analyze import analyze_dataset #Import analyze.py . check the analyze.py
from prepro import DataPreprocessor
import regression

dataset = pd.read_csv("bigmart.csv") #ّّImport dataset for example bigmart_sale "bigmart.csv" 

# analyze_dataset(dataset)  #It does as the name suggests 

preprocess = DataPreprocessor()

prepro_dataset = preprocess.object_to_numeric(dataset) #It does as the name suggests
prepro_dataset = preprocess.handle_null_values(prepro_dataset,-1) #Check the null values are not null values 
prepro_dataset = preprocess.standardize_dataset(prepro_dataset) #Standardization the dataset     
prepro_dataset = preprocess.normalize_dataset(prepro_dataset) #Normalaization.

# analyze_dataset(prepro_dataset)

X = prepro_dataset.iloc[:, :-1].values
y = prepro_dataset.iloc[:,1].values

# Create a RegressionModel instance and pass the dataset to it
reg_model = regression.RegressionModel(X, y, test_size=0.2, random_state=42)

# Linear Regression
coefficients_linear, intercept_linear, r2_score_linear = reg_model.perform_linear_regression()
print("Linear Regression Coefficients:", coefficients_linear)
print("Linear Regression Intercept:", intercept_linear)
print("Linear Regression R-squared Score:", r2_score_linear)


# Random Forest Regression
feature_importances, r2_score_rf = reg_model.perform_random_forest_regression()
print("Random Forest Regression Feature Importances:", feature_importances)
print("Random Forest Regression R-squared Score:", r2_score_rf)