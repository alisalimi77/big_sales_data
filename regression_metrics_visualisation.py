import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MetricsBuilder:
    def __init__(self, model_names, y_true, y_preds):
        self.model_names = model_names
        self.y_true = y_true
        self.y_preds = y_preds
        self.metrics_data = {
            'Model': self.model_names,
            'Mean Squared Error': [],
            'Mean Absolute Error': [],
            'Root Mean Squared Error': [],
            'R-squared': []
        }

    def calculate_metrics(self):
        for y_pred, model_name in zip(self.y_preds, self.model_names):
            mse = mean_squared_error(self.y_true, y_pred)
            mae = mean_absolute_error(self.y_true, y_pred)
            rmse = mean_squared_error(self.y_true, y_pred, squared=False)
            r2 = r2_score(self.y_true, y_pred)

            self.metrics_data['Mean Squared Error'].append(mse)
            self.metrics_data['Mean Absolute Error'].append(mae)
            self.metrics_data['Root Mean Squared Error'].append(rmse)
            self.metrics_data['R-squared'].append(r2)

        metrics_df = pd.DataFrame(self.metrics_data)
        return metrics_df
