from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class KNNRegressionModel:
    def __init__(self, n_neighbors=5):
        """
        Create a K-Nearest Neighbors Regression model.

        Parameters:
            n_neighbors (int): The number of neighbors to consider (default is 5).
        """
        self.n_neighbors = n_neighbors
        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)

    def train(self, X_train, y_train):
        """
        Train the K-Nearest Neighbors Regression model.

        Parameters:
            X_train (numpy.ndarray): Training feature data.
            y_train (numpy.ndarray): Training target data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the K-Nearest Neighbors Regression model.

        Parameters:
            X_test (numpy.ndarray): Test feature data.

        Returns:
            numpy.ndarray: Predicted target data.
        """
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, X_test, y_test):
        """
        Evaluate the K-Nearest Neighbors Regression model on the test set.

        Parameters:
            X_test (numpy.ndarray): Test feature data.
            y_test (numpy.ndarray): Test target data.

        Returns:
            float: Mean Squared Error (MSE)
            float: Mean Absolute Error (MAE)
            float: Root Mean Squared Error (RMSE)
            float: R-squared (R2)
        """
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        return mse, mae, rmse, r2
