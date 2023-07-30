from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score


class RandomForestModel:
    def __init__(self):
        """
        Create a Random Forest Classifier.

        Attributes:
            model (RandomForestClassifier): The Random Forest Classifier model.
        """
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X_train, y_train):
        """
        Train the Random Forest Classifier.

        Parameters:
            X_train (numpy.ndarray): Training feature data.
            y_train (numpy.ndarray): Training target data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the Random Forest Classifier.

        Parameters:
            X_test (numpy.ndarray): Test feature data.

        Returns:
            numpy.ndarray: Predicted target data.
        """
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, X_test, y_test):
        """
        Evaluate the Random Forest Classifier model on the test set.

        Parameters:
            X_test (numpy.ndarray): Test feature data.
            y_test (numpy.ndarray): Test target data.

        Returns:
            float: F1 Score
            float: Accuracy
        """
        y_pred = self.model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        return f1, accuracy
