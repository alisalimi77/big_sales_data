
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class RegressionModel:
      
        
    def train_linear_regression(self, X_train, y_train):
        # Create and train a Linear Regression model
        # LinearRegression implements ordinary least squares linear regression
        self.regression_model = LinearRegression() 
        self.regression_model.fit(X_train, y_train)

    def train_decision_tree_regression(self, X_train, y_train):
        # Create and train a Decision Tree Regression model
        # DecisionTreeRegressor implements a regression tree
        self.regression_model = DecisionTreeRegressor()
        self.regression_model.fit(X_train, y_train)

    def train_random_forest_regression(self, X_train, y_train):
        # Create and train a Random Forest Regression model
        # RandomForestRegressor trains an ensemble of decision trees
        self.regression_model = RandomForestRegressor()
        self.regression_model.fit(X_train, y_train)

    def test_regression_model(self, X_test):
        # Make predictions on the test set using the trained model
        y_pred = self.regression_model.predict(X_test)
        return y_pred

   # Add method to run full test and return metrics
    def test_and_evaluate(self, X_test, y_test):
        y_pred = self.test_regression_model(X_test)
        mse, mae, rmse, r2 = self.calculate_regression_metrics(y_test, y_pred)

        return mse, mae, rmse, r2
    def calculate_regression_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mse, mae, rmse, r2 

      