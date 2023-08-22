import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , GridSearchCV
import analyze 
from prepro import DataPreprocessor
from regression import RegressionModel 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prettytable import PrettyTable


# Load dataset
dataset = pd.read_csv("bigmart.csv")

# Analyze data if you need it 

analyze.analyze_dataset(dataset)

# Initialize DataPreprocessor
preprocessor = DataPreprocessor()

# Preprocess data
preprocessed_dataset = preprocessor.object_to_numeric(dataset)

print(preprocessed_dataset.head())

# check null valeus
print("Null Value Analysis:\n")
analyze.null_values(preprocessed_dataset)

# Handle null values

preprocessed_dataset = preprocessor.handle_null_values(preprocessed_dataset,'median')

# check null values
print("Null Value Analysis after handeling :\n")
analyze.null_values(preprocessed_dataset)




preprocessed_dataset = preprocessor.normalize_dataset(preprocessed_dataset)
preprocessed_dataset = preprocessor.standardize_dataset(preprocessed_dataset)


# Split data into features (X) and target (y)
X = preprocessed_dataset.iloc[:, :-1].values
y = preprocessed_dataset.iloc[:, -1].values



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)


print('X_train shape is :\n' , X_train.shape)
print('X_test shape is :\n' , X_test.shape)
print('y_train shape is :\n' , y_train.shape)
print('y_test shape is :\n' , y_test.shape)
print("\n\n")


# Initialize separate RegressionModel instances for each model

regressor_linear = RegressionModel()

regressor_decision_tree = RegressionModel()

regressor_random_forest = RegressionModel()

OrthogonalMatchingPursuit = RegressionModel()

ElasticNet = RegressionModel()

KNeighborsRegressor = RegressionModel()

PassiveAggressiveRegressor = RegressionModel()

ExtraTreesRegressor = RegressionModel()

XGBRegressor = RegressionModel()

HuberRegressor = RegressionModel()

LGBMRegressor = RegressionModel()

CatBoostRegressor = RegressionModel()

GradientBoostingRegressor = RegressionModel()

# Train models
regressor_linear.train_linear_regression(X_train, y_train)

regressor_decision_tree.train_decision_tree_regression(X_train, y_train)

regressor_random_forest.train_random_forest_regression(X_train, y_train)

OrthogonalMatchingPursuit.train_orthogonal_matching_pursuit(X_train, y_train)

ElasticNet.train_elastic_net_regression(X_train, y_train)

KNeighborsRegressor.train_knn_regression(X_train, y_train)

PassiveAggressiveRegressor.train_passive_aggressive_regression(X_train, y_train)

ExtraTreesRegressor.train_extra_trees_regression(X_train, y_train)

XGBRegressor.train_xgboost_regression(X_train, y_train)

HuberRegressor.train_huber_regression(X_train, y_train)

LGBMRegressor.train_lightgbm_regression(X_train, y_train)

CatBoostRegressor.train_catboost_regression(X_train, y_train)

GradientBoostingRegressor.train_gradient_boosting_regression(X_train, y_train)



# Evaluate each model

models = {
    "Linear Regression": regressor_linear,
    
    "Decision Tree Regression": regressor_decision_tree,
    
    "Random Forest Regression": regressor_random_forest,
    
    "Orthogonal Matching Pursuit": OrthogonalMatchingPursuit,
    
    "Elastic Net": ElasticNet,
    
    "K-Nearest Neighbors": KNeighborsRegressor,
    
    "Passive Aggressive Regressor": PassiveAggressiveRegressor,
    
    "Extra Trees Regressor": ExtraTreesRegressor,
    
    "Extreme Gradient Boosting": XGBRegressor,
    
    "Huber Regressor": HuberRegressor,
    
    "Light Gradient Boosting Machine": LGBMRegressor,
    
    "CatBoost Regressor": CatBoostRegressor,
    
    "Gradient Boosting Regressor": GradientBoostingRegressor


}



metrics_dict = {}

for name, model in models.items():
    if hasattr(model, 'train_' + name.lower().replace(" ", "_")):
        train_method = getattr(model, 'train_' + name.lower().replace(" ", "_"))
        train_method(X_train, y_train)
    y_pred = model.test_regression_model(X_test)
    mse, mae, rmse, r2 = model.calculate_regression_metrics(y_test, y_pred)
    metrics_dict[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R-squared': r2
    }

# Create a PrettyTable for displaying comparison table
table = PrettyTable()
table.field_names = ["Model", "RMSE", "MAE", "R-squared"]

for name, metrics in metrics_dict.items():
    table.add_row([name, metrics['RMSE'], metrics['MAE'], metrics['R-squared']])

# Print the comparison table
print(table)