import pandas as pd
from sklearn.model_selection import train_test_split
from analyze import analyze_dataset
from prepro import DataPreprocessor
from regression import RegressionModel

# Load dataset
dataset = pd.read_csv("bigmart.csv")

# Analyze dataset (if needed)
# analyze_dataset(dataset)

# Initialize DataPreprocessor
preprocessor = DataPreprocessor()

# Preprocess data
preprocessed_dataset = preprocessor.object_to_numeric(dataset)
preprocessed_dataset = preprocessor.handle_null_values(preprocessed_dataset, -1)
preprocessed_dataset = preprocessor.standardize_dataset(preprocessed_dataset)
preprocessed_dataset = preprocessor.normalize_dataset(preprocessed_dataset)

# Split data into features (X) and target (y)
X = preprocessed_dataset.iloc[:, :-1].values
y = preprocessed_dataset.iloc[:, -1].values

# Perform feature selection
# X_selected = preprocessor.feature_selection(X, y, k=10)  # Adjust k as needed

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize separate RegressionModel instances for each model
regressor_linear = RegressionModel()
regressor_decision_tree = RegressionModel()
regressor_random_forest = RegressionModel()

# Train models
regressor_linear.train_linear_regression(X_train, y_train)
regressor_decision_tree.train_decision_tree_regression(X_train, y_train)
regressor_random_forest.train_random_forest_regression(X_train, y_train)

# Evaluate models
models = {
    "Linear Regression": regressor_linear,
    "Decision Tree Regression": regressor_decision_tree,
    "Random Forest Regression": regressor_random_forest
}

for name, model in models.items():
    y_pred = model.test_regression_model(X_test)
    mse, mae, rmse, r2 = model.calculate_regression_metrics(y_test, y_pred)

    print(f"Evaluating {name} model:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R-squared:", r2)
    print()

print("Done")
