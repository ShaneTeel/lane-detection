class MinMaxScaler():

    def __init__(self):
        self.X_max = None
        self.X_min = None
        self.y_max = None
        self.y_min = None

    def transform(self, X, y):
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        
        self.X_min, self.X_max = X.min(), X.max()
        X = (X - self.X_min) / (self.X_max - self.X_min) if self.X_max > self.X_min else X

        self.y_min, self.y_max = y.min(), y.max()
        y = (y - self.y_min) / (self.y_max - self.y_min) if self.y_max > self.y_min else y

        return X, y

    def inverse_transform(self, X_scaled, y_scaled):
        X = X_scaled * (self.X_max - self.X_min) + self.X_min
        y = y_scaled * (self.y_max - self.y_min) + self.y_min
        return X, y