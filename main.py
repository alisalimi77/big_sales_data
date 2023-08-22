import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , GridSearchCV
from analyze import regression_model_analysis
from prepro import DataPreprocessor
from regression import RegressionModel 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prettytable import PrettyTable


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

fig, ax = plt.subplots(figsize=(6,6))

# Countplot 
sns.countplot(
    data=dataset,
    x='Outlet_Establishment_Year',
    ax=ax
)

# Labels and title
ax.set_ylabel('Frequency')
ax.set_xlabel('Establishment Year')
ax.set_title('Outlet Establishment Year Distribution')

# Display plot
plt.tight_layout()
plt.show()
# Initialize DataPreprocessor
preprocessor = DataPreprocessor()

# Preprocess data
preprocessed_dataset = preprocessor.object_to_numeric(dataset)
print(preprocessed_dataset.head())
preprocessed_dataset = preprocessor.handle_null_values(preprocessed_dataset,'median')
# Calculate null values
null_counts = preprocessed_dataset.isnull().sum()

# Create dataframe with column names
null_df = pd.DataFrame(null_counts, columns=['Number of Nulls'])  

# Calculate null percentage
total_cells = np.product(preprocessed_dataset.shape)
null_df['Percent Null'] = null_df['Number of Nulls'] / total_cells * 100

# Display dataframe
print("Null Value Analysis:\n")
print(null_df)
preprocessed_dataset = preprocessor.normalize_dataset(preprocessed_dataset)
preprocessed_dataset = preprocessor.standardize_dataset(preprocessed_dataset)


# Split data into features (X) and target (y)
X = preprocessed_dataset.iloc[:, :-1].values
y = preprocessed_dataset.iloc[:, -1].values



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)


print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)
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

# Print trained models
for name, model in models.items():
    print(f"{name} Model Instance: {model.regression_model}")


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