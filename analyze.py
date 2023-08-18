# analyze.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_regression_actual_vs_predicted(y_test, y_pred):
   plt.scatter(y_test, y_pred)
   plt.xlabel('Actual Values')
   plt.ylabel('Predicted Values')
   plt.title('Actual vs Predicted Values')
   plt.show()

def plot_residuals(y_test, y_pred):
   residuals = y_test - y_pred
   sns.displot(residuals)
   plt.title('Residuals Distribution')
   plt.show()

def regression_model_analysis(y_test, y_pred):
   plot_regression_actual_vs_predicted(y_test, y_pred)
   plot_residuals(y_test, y_pred)