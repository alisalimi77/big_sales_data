import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_dataset(dataset):
    """
    Analyze the dataset by displaying various information, statistics, and visualizations.

    Parameters:
        dataset (pandas.DataFrame): The input dataset to be analyzed.
    """
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

    # Visualize missing values
    plt.figure(figsize=(10, 6))
    missing_values.plot(kind="bar")
    plt.title("Missing Values")
    plt.xlabel("Columns")
    plt.ylabel("Count")
    plt.show()

    # Visualize distribution of numeric columns
    numeric_columns = dataset.select_dtypes(include=np.number).columns
    for column in numeric_columns:
        plt.figure(figsize=(8, 6))
        dataset[column].plot(kind="hist")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()
