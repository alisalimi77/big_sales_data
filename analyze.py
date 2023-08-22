import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_dataset(dataset):
    # Display dataset head
    print(dataset.head())
    # Display dataset information
    print(dataset.info())
    # Display descriptive statistics
    print(dataset.describe().transpose())

    # Check for null values
    null_counts = dataset.isnull().sum()
    # Create a dataframe to show null statistics
    null_df = pd.DataFrame(null_counts, columns=['Number of Nulls'])
    total_cells = np.product(dataset.shape)
    null_df['Percent Null'] = null_df['Number of Nulls'] / total_cells * 100
    # Display null statistics
    print("Null Value Analysis:\n")
    print(null_df)

    # Show duplicate data
    print(dataset[dataset.duplicated()])

    # Data Visualization
    plt.figure(figsize=(10, 20))
    for i, col in enumerate(['Outlet_Size', 'Outlet_Location_Type', 'Item_Fat_Content'], 1):
        plt.subplot(3, 2, i)
        sns.countplot(x=col, data=dataset, palette='Set2')
        plt.subplot(3, 2, i + 1)
        sns.countplot(x=col, hue='Outlet_Type', data=dataset, palette='Set2')
    plt.tight_layout()
    plt.show()

    # Countplot for establishment year distribution
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.countplot(data=dataset, x='Outlet_Establishment_Year', ax=ax)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Establishment Year')
    ax.set_title('Outlet Establishment Year Distribution')
    plt.tight_layout()
    plt.show()


def plot_regression_actual_vs_predicted(y_test, y_pred):
   plt.scatter(y_test, y_pred)
   plt.xlabel('Actual Values')
   plt.ylabel('Predicted Values')
   plt.title('Actual vs Predicted Values')
   plt.show()

def plot_residuals(y_test, y_pred):
   residuals = y_test - y_pred
   sns.displot(residuals)
   plt.title('Residuals Distribution')
   plt.show()

def regression_model_analysis(y_test, y_pred):
   plot_regression_actual_vs_predicted(y_test, y_pred)
   plot_residuals(y_test, y_pred)


def null_values(dataset):
    # Calculate null values
    null_counts = dataset.isnull().sum()

    # Create dataframe with column names
    null_df = pd.DataFrame(null_counts, columns=['Number of Nulls'])  

    # Calculate null percentage
    total_cells = np.product(dataset.shape)
    null_df['Percent Null'] = null_df['Number of Nulls'] / total_cells * 100

    # Display dataframe
    print("Null Value Analysis:\n")
    print(null_df)