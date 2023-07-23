from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class RegressionModel:
    def __init__(self, X, y, test_size=0.2, random_state=None):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split_data()

    def train_test_split_data(self):
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def perform_linear_regression(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        return model.coef_, model.intercept_, model.score(self.X_test, self.y_test)

    def perform_logistic_regression(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        return model.coef_, model.intercept_, model.score(self.X_test, self.y_test)

    def perform_random_forest_regression(self):
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        return model.feature_importances_, model.score(self.X_test, self.y_test)
