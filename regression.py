from sklearn.linear_model import LinearRegression, LogisticRegression , SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

class RegressionModel:
    def __init__(self, X, y, test_size=0.2, random_state=None):
        """
        Initialize the RegressionModel with the dataset and other parameters.

        Parameters:
        - X: numpy array or pandas DataFrame
            The feature data (independent variables).
        - y: numpy array or pandas Series
            The target data (dependent variable).
        - test_size: float (default: 0.2)
            The proportion of the dataset to include in the test split.
        - random_state: int or None (default: None)
            Controls the randomness of the train-test split.

        Initializes the training and testing datasets based on the provided parameters.
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split_data()

    def train_test_split_data(self):
        """
        Split the dataset into training and testing sets.

        Returns:
        - X_train: numpy array or pandas DataFrame
            The training feature data.
        - X_test: numpy array or pandas DataFrame
            The testing feature data.
        - y_train: numpy array or pandas Series
            The training target data.
        - y_test: numpy array or pandas Series
            The testing target data.

        Splits the dataset into training and testing sets based on the test_size and random_state.
        """
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
    
    def perform_linear_regression(self):
        """
        Perform linear regression on the dataset.

        Returns:
        - coefficients: numpy array
            The coefficients of the linear regression model.
        - intercept: float
            The intercept of the linear regression model.
        - r2_score: float
            The R-squared score of the linear regression model.

        Fits a LinearRegression model to the training data and returns the coefficients,
        intercept, and R-squared score on the testing data.
        """
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        return model.coef_, model.intercept_, model.score(self.X_test, self.y_test)

    def perform_logistic_regression(self):
        """
        Perform logistic regression on the dataset.

        Returns:
        - coefficients: numpy array
            The coefficients of the logistic regression model.
        - intercept: float
            The intercept of the logistic regression model.
        - accuracy: float
            The accuracy of the logistic regression model.
        - f1_score: float
            The F1 score of the logistic regression model.

        Fits a LogisticRegression model to the training data and returns the coefficients,
        intercept, accuracy, and F1 score on the testing data.
        """
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        accuracy = accuracy_score(self.y_test, model.predict(self.X_test))
        f1 = f1_score(self.y_test, model.predict(self.X_test))
        return model.coef_, model.intercept_, accuracy, f1

    def perform_random_forest_regression(self):
        """
        Perform random forest regression on the dataset.

        Returns:
        - feature_importances: numpy array
            The feature importances from the random forest model.
        - r2_score: float
            The R-squared score of the random forest model.

        Fits a RandomForestRegressor model to the training data and returns the feature importances
        and R-squared score on the testing data.
        """
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        return model.feature_importances_, model.score(self.X_test, self.y_test)

    def perform_sgd_regression(self):
        """
        Perform Stochastic Gradient Descent (SGD) regression on the dataset.

        Returns:
        - coefficients: numpy array
            The coefficients of the SGD regression model.
        - intercept: float
            The intercept of the SGD regression model.
        - r2_score: float
            The R-squared score of the SGD regression model.

        Fits an SGDRegressor model to the training data and returns the coefficients,
        intercept, and R-squared score on the testing data.
        """
        model = SGDRegressor()
        model.fit(self.X_train, self.y_train)
        return model.coef_, model.intercept_, model.score(self.X_test, self.y_test)
