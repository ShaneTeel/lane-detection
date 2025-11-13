import numpy as np

class RegressionMetrics():

    def __init__(self, name:str=None):

        self.r2 = []
        self.mse = []
        self.rmse = []
        self.mae = []
        self.name = "Regression" if name is None else name
        self.report = None

    def compute_metrics(self, y_true, y_pred):
        n = len(y_true)
        if n == 0:
            print("`y_true` contains no points.")
            return
        
        self.r2.append(self._get_r2(y_true, y_pred))
        self.mse.append(self._get_mse(y_true, y_pred, n))
        self.rmse.append(self._get_rmse(y_true, y_pred, n))
        self.mae.append(self._get_mae(y_true, y_pred))
        
    def _get_rss(self, y_true, y_pred):
        return np.sum((y_true - y_pred)**2)
    
    def _get_tss(self, y_true):
        y_mean = np.mean(y_true)
        return np.sum((y_true - y_mean)**2)

    def _get_r2(self, y_true, y_pred):
        rss = self._get_rss(y_true, y_pred)
        tss = self._get_tss(y_true)
        r2 = 1 - rss / tss
        return r2
    
    def _get_mse(self, y_true, y_pred, n):
        rss = self._get_rss(y_true, y_pred)
        return rss / n
    
    def _get_rmse(self, y_true, y_pred, n):
        mse = self._get_mse(y_true, y_pred, n)
        return mse**0.5
    
    def _get_mae(self, y_true, y_pred):
        return np.mean(abs(y_true - y_pred))
    
    def return_metrics(self):
        return {"R2": np.mean(self.r2), "MSE": np.mean(self.mse), "RMSE": np.mean(self.rmse), "MAE": np.mean(self.mae)}