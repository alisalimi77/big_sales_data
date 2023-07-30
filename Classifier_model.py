from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score


def create_random_forest_classifier():
    """
    Create a Random Forest Classifier.

    Returns:
        RandomForestClassifier: The Random Forest Classifier model.
    """
    rf_classifier = RandomForestClassifier(random_state=42)
    return rf_classifier


def train_random_forest_classifier(model, X_train, y_train):
    """
    Train the Random Forest Classifier.

    Parameters:
        model (RandomForestClassifier): The Random Forest Classifier model.
        X_train (numpy.ndarray): Training feature data.
        y_train (numpy.ndarray): Training target data.
    """
    model.fit(X_train, y_train)


def predict_random_forest_classifier(model, X_test):
    """
    Make predictions using the Random Forest Classifier.

    Parameters:
        model (RandomForestClassifier): The trained Random Forest Classifier model.
        X_test (numpy.ndarray): Test feature data.

    Returns:
        numpy.ndarray: Predicted target data.
    """
    y_pred = model.predict(X_test)
    return y_pred


def evaluate_random_forest_classifier(model, X_test, y_test):
    """
    Evaluate the Random Forest Classifier model on the test set.

    Parameters:
        model (RandomForestClassifier): The trained Random Forest Classifier model.
        X_test (numpy.ndarray): Test feature data.
        y_test (numpy.ndarray): Test target data.

    Returns:
        float: F1 Score
        float: Accuracy
    """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    return f1, accuracy
