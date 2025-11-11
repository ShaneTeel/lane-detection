class Evaluator():

    def __init__(self):

        self.metrics = {
            "left": {"mse": 0.0, "rmse": 0.0, "r2": 0.0},
            "right": {"mse": 0.0, "rmse": 0.0, "r2": 0.0}
        }

    def gen_report(self):
        pass

    def evaluate(self, y_true, y_pred, direction):
        n = len(y_true)
        r2, rss = self._get_r2(y_true, y_pred, n)
        self.metrics[direction]["r2"] += r2

        mse = self._get_mse(n, rss)
        self.metrics[direction]["mse"] += mse

        rmse = self._get_rmse(mse)
        self.metrics[direction]["rmse"] += rmse

    def _get_rss(self, y_true, y_pred):
        return sum((y_true - y_pred)**2)
    
    def _get_tss(self, y_true, n):
        y_mean = sum(y_true) / n
        return sum((y_true - y_mean)**2)

    def _get_r2(self, y_true, y_pred, n):
        rss = self._get_rss(y_true, y_pred)
        tss = self._get_tss(y_true, n)
        r2 = 1 - rss / tss
        return r2, rss
    
    def _get_mse(self, n, rss):
        return rss / n
    
    def _get_rmse(self, mse):
        return mse**0.5
    
