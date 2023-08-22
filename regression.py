from sklearn.linear_model import (
    LinearRegression,
    OrthogonalMatchingPursuit,
    ElasticNet,
    PassiveAggressiveRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import HuberRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class RegressionModel:

    def __init__(self, regression_model=None):
        self.regression_model = regression_model
        
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
        
    def train_orthogonal_matching_pursuit(self, X_train, y_train):
        # Create and train an Orthogonal Matching Pursuit model
        self.regression_model = OrthogonalMatchingPursuit()
        self.regression_model.fit(X_train, y_train)

    def train_elastic_net_regression(self, X_train, y_train):
        # Create and train an Elastic Net Regression model
        self.regression_model = ElasticNet()
        self.regression_model.fit(X_train, y_train)

    def train_knn_regression(self, X_train, y_train):
        # Create and train a k-nearest neighbors regression model
        self.regression_model = KNeighborsRegressor()
        self.regression_model.fit(X_train, y_train)

    def train_passive_aggressive_regression(self, X_train, y_train):
        # Create and train a Passive Aggressive Regressor model
        self.regression_model = PassiveAggressiveRegressor()
        self.regression_model.fit(X_train, y_train)

    def train_extra_trees_regression(self, X_train, y_train):
        # Create and train an Extra Trees Regressor model
        self.regression_model = ExtraTreesRegressor()
        self.regression_model.fit(X_train, y_train)

    def train_xgboost_regression(self, X_train, y_train):
        # Create and train an Extreme Gradient Boosting (XGBoost) Regressor model
        self.regression_model = XGBRegressor()
        self.regression_model.fit(X_train, y_train)

    def train_huber_regression(self, X_train, y_train):
        # Create and train a Huber Regressor model
        self.regression_model = HuberRegressor()
        self.regression_model.fit(X_train, y_train)

    def train_lightgbm_regression(self, X_train, y_train):
        # Create and train a Light Gradient Boosting Machine (LightGBM) Regressor model
        self.regression_model = LGBMRegressor()
        self.regression_model.fit(X_train, y_train)

    def train_catboost_regression(self, X_train, y_train):
        # Create and train a CatBoost Regressor model
        self.regression_model = CatBoostRegressor()
        self.regression_model.fit(X_train, y_train)

    def train_gradient_boosting_regression(self, X_train, y_train):
        # Create and train a Gradient Boosting Regressor model
        self.regression_model = GradientBoostingRegressor()
        self.regression_model.fit(X_train, y_train)

    def test_regression_model(self, X_test):
        if self.regression_model is not None:
            y_pred = self.regression_model.predict(X_test)
            return y_pred
        else:
            raise ValueError("Regression model is not trained yet. Call a train method first.")

   # Add method to run full test and return metrics
    # def test_and_evaluate(self, X_test, y_test):
    #     y_pred = self.test_regression_model(X_test)
    #     mse, mae, rmse, r2 = self.calculate_regression_metrics(y_test, y_pred)

    #     return mse, mae, rmse, r2
    def calculate_regression_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mse, mae, rmse, r2
      