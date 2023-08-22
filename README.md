# Bigmart Sales Prediction

This is a Python script to predict sales for the Bigmart retail dataset. The script uses a range of regression techniques, including Linear Regression, Decision Tree Regression, and Random Forest Regression. It also incorporates advanced data preprocessing techniques.

The main.py file contains the primary execution script.

## Getting Started

### Prerequisites

To run this script, you'll need the following libraries installed:

- pandas
- seaborn
- matplotlib
- scikit-learn
- numpy
- prettytable

You can install these libraries using the following command:

```bash
pip install pandas seaborn matplotlib scikit-learn numpy prettytable
```

### Usage
1. Clone this repository to your local machine:
- https://github.com/alisalimi77/big_sales_data.git

2. Navigate to the repository directory:

- To run the script, simply execute the `main.py` file:


## Dataset

The dataset used in this script is `bigmart.csv`, which is a file containing sales data from a range of Bigmart retail stores. The dataset should be in the same directory as the script.

## Structure

- `main.py`: The main script that loads the dataset, preprocesses the data, splits it into training and testing sets, trains various regression models, tests the models, and finally displays various regression metrics for each model.

- `analyze.py`: This file contains the function `analyze_dataset()` that can be used to analyze any given dataset.

- `prepro.py`: This file contains the `DataPreprocessor` class with various methods for preprocessing data, including handling null values, standardizing the dataset, and normalizing the dataset.

- `regression.py`: This file contains functions for training and testing various regression models, as well as calculating and displaying regression metrics.

## Key Functions

Key Functions
This section outlines the key functions used in the Bigmart Sales Prediction project.

## Data Preprocessing
`DataPreprocessor.object_to_numeric(dataset)`: Converts object data types to numeric for improved compatibility with machine learning algorithms.

`DataPreprocessor.handle_null_values(dataset, replacement_value)`: Detects and handles null values in the dataset by replacing them with the specified replacement value.

`DataPreprocessor.standardize_dataset(dataset)`: Standardizes the dataset by scaling the features to have a mean of 0 and a standard deviation of 1.

`DataPreprocessor.normalize_dataset(dataset)`: Normalizes the dataset by scaling the features to a specified range, typically between 0 and 1.

## Regression Models

Each regression model has the following common methods:

`train_model(X_train, y_train)`: Trains the regression model using the provided training data.

`test_regression_model(X_test)`: Predicts the target values for the given test features using the trained model.

`calculate_regression_metrics(y_true, y_pred)`: Calculates and returns a set of common regression metrics, including Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared.

### The following regression models are available:

Linear Regression

Decision Tree Regression

Random Forest Regression

Orthogonal Matching Pursuit

Elastic Net

K-Nearest Neighbors Regression

Passive Aggressive Regression

Extra Trees Regression

Extreme Gradient Boosting (XGBoost) Regression

Huber Regression

Light Gradient Boosting Machine (LGBM) Regression

CatBoost Regression

Gradient Boosting Regression

Model Evaluation and Comparison

### The main.py script provides the following workflow for model evaluation and comparison:

- Load and preprocess the dataset
- Split the data into training and testing sets.
- Train each regression model using the training data.
- Evaluate each model using the testing data.
- Calculate and display regression metrics for each model.
- A comparison table is generated using the PrettyTable library, displaying the RMSE, MAE, and R-squared metrics for each model.

## Contributing

We welcome any contributions to this project. Please feel free to submit a pull request or open an issue for any changes or improvements you would like to suggest. 
- Ali Salimi https://github.com/alisalimi77
- Sahar Radmehr https://github.com/Sahar-rad
## License

This project is licensed under the MIT License. 

## Contact

If you have any questions or need further clarification about the project, feel free to reach out.
