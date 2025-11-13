class MinMaxScaler():

    def __init__(self):
        self._is_fitted = False

    def fit(self, X, y):
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        
        self.X_min, self.X_max = X.min(), X.max()
        self.y_min, self.y_max = y.min(), y.max()
        self._is_fitted = True
        return self

    def transform(self, X, y):
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        X = (X - self.X_min) / (self.X_max - self.X_min) if self.X_max > self.X_min else X
        y = (y - self.y_min) / (self.y_max - self.y_min) if self.y_max > self.y_min else y

        return X, y
    
    def fit_transform(self, X, y):
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y):
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")

        X = X * (self.X_max - self.X_min) + self.X_min
        y = y * (self.y_max - self.y_min) + self.y_min
        return X, y
    
    def _inverse_transform_X(self, X):
        if X is None:
            raise ValueError(f"Error: 'X' ({X}) == 'NoneType'")
        
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        
        return X * (self.X_max - self.X_min) + self.X_min
    
    def _inverse_transform_y(self, y):
        if y is None:
            raise ValueError(f"Error: 'y' ({y}) == 'NoneType'")
        
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        
        return y * (self.y_max - self.y_min) + self.y_min
    
    def _transform_X(self, X):
        if X is None:
            raise ValueError(f"Error: 'X' ({X}) == 'NoneType'")
        
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        return (X - self.X_min) / (self.X_max - self.X_min) if self.X_max > self.X_min else X
    
    def _transform_y(self, y):
        if y is None:
            raise ValueError(f"Error: 'y' ({y}) == 'NoneType'")
        
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        
        return (y - self.y_min) / (self.y_max - self.y_min) if self.y_max > self.y_min else y