
import pandas as pd 
import numpy as np 

def analyze_dataset(dataset):
    # Display the shape of the dataset
    print("------Data Shape-----\n\n", dataset.shape)

    # Display the first few rows of the dataset
    print("\n\n-----Data head----- \n\n", dataset.head())

    # Display information about the dataset
    print("\n\n\n-----Data information-----\n")
    print(dataset.info())

    # Display the size of the dataset
    print("\n\n\n Data size :>>>>> ", dataset.size)

    # Display the column names of the dataset
    print("\n\n-----Column names----\n\n", dataset.columns)

    # Display descriptive statistics of the dataset
    print("\n\n-----Descriptive statistics----\n\n")
    print(dataset.describe(include="all"))

    # Count the number of missing values in each column
    missing_values = dataset.isnull().sum()
    print("\n-----Missing Values-----\n", missing_values)