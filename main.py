import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , GridSearchCV
from analyze import regression_model_analysis
from prepro import DataPreprocessor
from regression import RegressionModel 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# Load dataset
dataset = pd.read_csv("bigmart.csv")

# Analyze dataset (if needed)
# Data info & head
print(dataset.head())
print(dataset.info())
# Data describe
print(dataset.describe().transpose())

#null Data?
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

# Show Data contain Duplicate data
print(dataset[dataset.duplicated()])

#Data Visualization
plt.figure(figsize=(10,20)) 

# Loop through columns to plot
for i, col in enumerate(['Outlet_Size', 'Outlet_Location_Type', 'Item_Fat_Content'], 1):

  # Subplot for count plot 
  plt.subplot(3,2,i)
  sns.countplot(x=col, data=dataset, palette='Set2')

  # Subplot for count plot by outlet type
  plt.subplot(3,2,i+1) 
  sns.countplot(x=col, hue='Outlet_Type', data=dataset, palette='Set2')

# Show plot  
plt.tight_layout()
plt.show()
# Initialize DataPreprocessor
preprocessor = DataPreprocessor()

# Preprocess data
preprocessed_dataset = preprocessor.object_to_numeric(dataset)
preprocessed_dataset = preprocessor.handle_null_values(preprocessed_dataset, -1)
preprocessed_dataset = preprocessor.normalize_dataset(preprocessed_dataset)
preprocessed_dataset = preprocessor.standardize_dataset(preprocessed_dataset)


# Split data into features (X) and target (y)
X = preprocessed_dataset.iloc[:, :-1].values
y = preprocessed_dataset.iloc[:, -1].values



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)

# Initialize separate RegressionModel instances for each model

# regressor_linear = RegressionModel()
# regressor_decision_tree = RegressionModel()
# regressor_random_forest = RegressionModel()

# # Train models
# regressor_linear.train_linear_regression(X_train, y_train)
# regressor_decision_tree.train_decision_tree_regression(X_train, y_train)
# regressor_random_forest.train_random_forest_regression(X_train, y_train)

# # Evaluate models
# models = {
#     "Linear Regression": regressor_linear,
#     "Decision Tree Regression": regressor_decision_tree,
#     "Random Forest Regression": regressor_random_forest
# }

# metrics = {}

# for name, model in models.items():
#    # Get metrics
#    mse, mae, rmse, r2 = model.test_and_evaluate(X_test, y_test)
   
#    metrics[name] = {
#       'mse': mse, 
#       'mae': mae,
#       'rmse': rmse,
#       'r2': r2
#    }

# # Select best model based on rmse
# best_model = min(metrics, key=lambda model: metrics[model]['rmse'])
# print("Best model based on RMSE:", best_model)

# # Tune hyperparameters of best model using GridSearchCV
# if best_model == "Random Forest Regression":
#    param_grid = {
#       'n_estimators': [100, 200, 500],
#       'max_depth': [5, 8, 15]
#    }

#    grid_search = GridSearchCV(models[best_model].regression_model, param_grid)
#    grid_search.fit(X_train, y_train)
   
#    print("Best hyperparameters:", grid_search.best_params_)
   
#    models[best_model].regression_model = grid_search.best_estimator_
#    # Evaluate each model
# for name, model in models.items():
#    y_pred = model.test_regression_model(X_test)
#    print(name, "performance:")
   
#    # Pass prediction to analysis
#    regression_model_analysis(y_test, y_pred)
#    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  
#    mae = mean_absolute_error(y_test, y_pred)
#    r2 = r2_score(y_test, y_pred)

#    print(name)
#    print("- RMSE: ", rmse)
#    print("- MAE: ", mae)
#    print("- R-squared: ", r2)

#    print("Done!!!")