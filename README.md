# Bigmart Sales Prediction

This is a Python script to predict sales for the Bigmart retail dataset. The script uses a range of regression techniques, including Linear Regression, Decision Tree Regression, and Random Forest Regression. It also incorporates advanced data preprocessing techniques.

The main.py file contains the primary execution script.

## Getting Started

### Prerequisites

To run this script, you'll need the following libraries installed:

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

You can install them using pip:


### Usage

To run the script, simply execute the `main.py` file:


## Dataset

The dataset used in this script is `bigmart.csv`, which is a file containing sales data from a range of Bigmart retail stores. The dataset should be in the same directory as the script.

## Structure

- `main.py`: The main script that loads the dataset, preprocesses the data, splits it into training and testing sets, trains various regression models, tests the models, and finally displays various regression metrics for each model.

- `analyze.py`: This file contains the function `analyze_dataset()` that can be used to analyze any given dataset.

- `prepro.py`: This file contains the `DataPreprocessor` class with various methods for preprocessing data, including handling null values, standardizing the dataset, and normalizing the dataset.

- `regression.py`: This file contains functions for training and testing various regression models, as well as calculating and displaying regression metrics.

## Key Functions

- `analyze_dataset(dataset)`: Analyzes the given dataset.

- `DataPreprocessor.object_to_numeric(dataset)`: Converts object data types to numeric.

- `DataPreprocessor.handle_null_values(dataset, replacement_value)`: Checks and handles null values in the dataset.

- `DataPreprocessor.standardize_dataset(dataset)`: Standardizes the dataset.

- `DataPreprocessor.normalize_dataset(dataset)`: Normalizes the dataset.

- `regression.train_linear_regression(X_train, y_train)`: Trains a linear regression model on the given training data.

- `regression.train_decision_tree_regression(X_train, y_train)`: Trains a decision tree regression model on the given training data.

- `regression.train_random_forest_regression(X_train, y_train)`: Trains a random forest regression model on the given training data.

- `regression.test_regression_model(model, X_test)`: Tests the given model on the testing data.

- `regression.calculate_regression_metrics(y_true, y_pred)`: Calculates a range of regression metrics.

- `display_metrics(regression_name, y_true, y_pred)`: Displays regression metrics for the given regression results.

## Contributing

We welcome any contributions to this project. Please feel free to submit a pull request or open an issue for any changes or improvements you would like to suggest. 

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

## Contact

If you have any questions or need further clarification about the project, feel free to reach out.
