class RegressionEvaluator():

    _BOLD = "\033[1m"
    _ITALICS = "\033[3m"
    _UNDERLINE = "\033[4m"
    _END = "\033[0m" 

    def __init__(self):

        self.left = None
        self.right = None

    def evaluate(self, y_true, y_pred, direction):
        if direction == "left" and self.left is None:
            self.left = RegressionMetrics()
        elif direction == "right" and self.right is None:
            self.right = RegressionMetrics()

        metrics = self.left if direction == "left" else self.right

        metrics.calc_metrics(y_true, y_pred)
    
    def regression_report(self, name):
        left_avgs = self._calc_avgs(self.left)
        right_avgs = self._calc_avgs(self.right)

        report_title = f"\n{self._BOLD}{self._UNDERLINE}{self._ITALICS}{name} Report{self._END}\n"
        report_header = f"{self._BOLD}Metrics{self._END}{'R2':>10} {'MSE':>10} {'RMSE':>10} {'MAE':>10}{self._END}\n"
        report_left = f"{self._ITALICS}{'Left':>7}{self._END} {left_avgs[0]:10.2f} {left_avgs[1]:10.2f} {left_avgs[2]:10.2f} {left_avgs[3]:10.2f} {self._END}\n"
        report_right = f"{self._ITALICS}{'Right':>7}{self._END} {right_avgs[0]:10.2f} {right_avgs[1]:10.2f} {right_avgs[2]:10.2f} {right_avgs[3]:10.2f} {self._END}\n"
        
        print(report_title)
        print(report_header)
        print(report_left)
        print(report_right)

    def _calc_avgs(self, attr):
        avgs = []
        for _, lst in attr.__dict__.items():
            avgs.append(sum(lst) / len(lst) if len(lst) != 0 else 0)
        return avgs

class RegressionMetrics():

    def __init__(self):

        self.r2 = []
        self.mse = []
        self.rmse = []
        self.mae = []

    def calc_metrics(self, y_true, y_pred):
        n = len(y_true)
        if n == 0:
            print("`y_true` contains no points.")
            return
        
        self.r2.append(self._get_r2(y_true, y_pred, n))
        self.mse.append(self._get_mse(y_true, y_pred, n))
        self.rmse.append(self._get_rmse(y_true, y_pred, n))
        self.mae.append(self._get_mae(y_true, y_pred, n))
        
    def _get_rss(self, y_true, y_pred):
        return sum((y_true - y_pred)**2)
    
    def _get_tss(self, y_true, n):
        y_mean = sum(y_true) / n
        return sum((y_true - y_mean)**2)

    def _get_r2(self, y_true, y_pred, n):
        rss = self._get_rss(y_true, y_pred)
        tss = self._get_tss(y_true, n)
        r2 = 1 - rss / tss
        return r2
    
    def _get_mse(self, y_true, y_pred, n):
        rss = self._get_rss(y_true, y_pred)
        return rss / n
    
    def _get_rmse(self, y_true, y_pred, n):
        mse = self._get_mse(y_true, y_pred, n)
        return mse**0.5
    
    def _get_mae(self, y_true, y_pred, n):
        return sum(abs(y_true - y_pred)) / n