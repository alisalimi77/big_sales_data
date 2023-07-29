
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from analyze import analyze_dataset
from prepro import DataPreprocessor
import regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# ّّImport dataset for example bigmart_sale "bigmart.csv"
dataset = pd.read_csv("bigmart.csv")

# analyze_dataset(dataset)  #It does as the name suggests

preprocess = DataPreprocessor()

prepro_dataset = preprocess.object_to_numeric(
    dataset)  # It does as the name suggests
prepro_dataset = preprocess.handle_null_values(
    prepro_dataset, -1)  # Check the null values are not null values
prepro_dataset = preprocess.standardize_dataset(
    prepro_dataset)  # Standardization the dataset
# Normalaization.
prepro_dataset = preprocess.normalize_dataset(prepro_dataset)

# analyze_dataset(prepro_dataset)

X = prepro_dataset.iloc[:, :-1].values
y = prepro_dataset.iloc[:, 1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train and test Linear Regression
linear_regression_model = regression.train_linear_regression(X_train, y_train)
y_pred_linear = regression.test_regression_model(
    linear_regression_model, X_test)

# Train and test Decision Tree Regression
decision_tree_regression_model = regression.train_decision_tree_regression(
    X_train, y_train)
y_pred_dt = regression.test_regression_model(
    decision_tree_regression_model, X_test)

# Train and test Random Forest Regression
random_forest_regression_model = regression.train_random_forest_regression(
    X_train, y_train)
y_pred_rf = regression.test_regression_model(
    random_forest_regression_model, X_test)

# Calculate and display regression metrics


def display_metrics(regression_name, y_true, y_pred):
    mse, mae, rmse, r2 = regression.calculate_regression_metrics(
        y_true, y_pred)
    print(f"{regression_name} Regression Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)
    print()


display_metrics("Linear", y_test, y_pred_linear)
display_metrics("Decision Tree", y_test, y_pred_dt)
display_metrics("Random Forest", y_test, y_pred_rf)
# Initialize the models
rf_classifier = RandomForestClassifier(random_state=42)

# Train the models
rf_classifier.fit(X_train, y_train)

# Predict on the validation set
y_val_pred_rf = rf_classifier.predict(X_test)

# Calculate metrics for the Random Forest Classifier
f1_score_rf = f1_score(y_test, y_val_pred_rf, average='weighted')
accuracy_rf = accuracy_score(y_test, y_val_pred_rf)

print("Random Forest Classifier - F1 Score:", f1_score_rf)
print("Random Forest Classifier - Accuracy:", accuracy_rf)
=======
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and test Linear Regression
linear_regression_model = regression.train_linear_regression(X_train, y_train)
y_pred_linear = regression.test_regression_model(linear_regression_model, X_test)

# Train and test Decision Tree Regression
decision_tree_regression_model = regression.train_decision_tree_regression(X_train, y_train)
y_pred_dt = regression.test_regression_model(decision_tree_regression_model, X_test)

# Train and test Random Forest Regression
random_forest_regression_model = regression.train_random_forest_regression(X_train, y_train)
y_pred_rf = regression.test_regression_model(random_forest_regression_model, X_test)

# Calculate and display regression metrics
def display_metrics(regression_name, y_true, y_pred):
    mse, mae, rmse, r2 = regression.calculate_regression_metrics(y_true, y_pred)
    print(f"{regression_name} Regression Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)
    print()

display_metrics("Linear", y_test, y_pred_linear)
display_metrics("Decision Tree", y_test, y_pred_dt)
display_metrics("Random Forest", y_test, y_pred_rf)

