# regression.py
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_linear_regression(X_train, y_train):
    # Create and train the Linear Regression model
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    return regression_model


def train_decision_tree_regression(X_train, y_train):
    # Create and train the Decision Tree Regression model
    regression_model = DecisionTreeRegressor()
    regression_model.fit(X_train, y_train)
    return regression_model


def train_random_forest_regression(X_train, y_train):
    # Create and train the Random Forest Regression model
    regression_model = RandomForestRegressor()
    regression_model.fit(X_train, y_train)
    return regression_model


def test_regression_model(regression_model, X_test):
    # Predict using the trained model
    y_pred = regression_model.predict(X_test)
    return y_pred


def calculate_regression_metrics(y_true, y_pred):
    # Calculate regression metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # Calculate RMSE from MSE
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, rmse, r2
