import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score


def tune_random_forest(X_train, y_train, param_grid, cv=5):
    """
    Tune hyperparameters of the Random Forest Classifier using Grid Search.

    Parameters:
        X_train (numpy.ndarray): Training feature data.
        y_train (numpy.ndarray): Training target data.
        param_grid (dict): Hyperparameter grid for Grid Search.
        cv (int, optional): Number of cross-validation folds. Default is 5.

    Returns:
        RandomForestClassifier: The tuned Random Forest Classifier model.
    """
    rf_classifier = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_classifier, param_grid=param_grid, scoring='f1_weighted', cv=cv)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    best_rf_classifier = RandomForestClassifier(random_state=42, **best_params)
    best_rf_classifier.fit(X_train, y_train)

    return best_rf_classifier


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the Random Forest Classifier model on the test set.

    Parameters:
        model (RandomForestClassifier): The trained Random Forest Classifier model.
        X_test (numpy.ndarray): Test feature data.
        y_test (numpy.ndarray): Test target data.
    """
    y_val_pred_rf = model.predict(X_test)

    f1_score_rf = f1_score(y_test, y_val_pred_rf, average='weighted')
    accuracy_rf = accuracy_score(y_test, y_val_pred_rf)

    print("Random Forest Classifier - F1 Score:", f1_score_rf)
    print("Random Forest Classifier - Accuracy:", accuracy_rf)
