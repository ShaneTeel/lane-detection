import numpy as np

class RANSACRegression():
    '''Test'''

    def __init__(self, degree:int = 2, n_iter:int=50, min_inliers:float = 0.8, max_error:int = 10):

        self.degree = degree
        self.poly_size = self.degree + 1
        self.sample_size = self.poly_size + 2
        self.n_iter = n_iter
        self.min_inliers = min_inliers
        self.max_error = max_error
        self.inlier_ratio = None

    def fit(self, X, y, max_error):

        # Create variables
        best_inliers = None
        best_inlier_count = 0
        best_coeffs = None
        population = len(X)

        # Determine consensus count. Support fraction (0<min_inliers<1) or absolute count (>=1).        
        if self.min_inliers is None:
            consensus = int(np.ceil(population * 0.5))
        else:
            try:
                if self.min_inliers < 1:
                    consensus = int(np.ceil(population * self.min_inliers))
                else:
                    consensus = int(self.min_inliers)
            except TypeError:
                # fallback to 50% if misconfigured
                consensus = int(np.ceil(population * 0.5))

        # If there are too few points to estimate the polynomial, fall back to OLS
        if population < self.poly_size:
            return self._calc_coeffs(X, y)

        # Use the minimal sample size necessary for the model (poly_size)
        sample_size = min(max(self.sample_size, 1), population)
        
        for _ in range(self.n_iter):
            # Random sampling
            sample_idx = np.random.choice(population, size=sample_size, replace=False)
            sample_X = X[sample_idx]
            sample_y = y[sample_idx]

            # Fit polynomial to samples
            try:
                coeffs = self._calc_coeffs(sample_X, sample_y)
                if not isinstance(coeffs, np.ndarray) or len(coeffs) != self.poly_size:
                    print(f"WARNING: calcualted coeffs not of correct type ({np.ndarray}) or correct size ({self.poly_size})")
                    continue

            except (np.linalg.LinAlgError, TypeError, ValueError) as e:
                print(f"WARNING: polyfit error - {e}")
                continue

            # Evaluate sample fit on all points (original scaled X in column 1)
            y_pred = self._poly_val(coeffs, X[:, 1])


            # Use absolute error
            sample_errors = np.abs(y - y_pred)

            # Count inliers (points close to fit)
            # Ensure max_error is a scalar; if None, treat as very small to avoid accepting everything
            if max_error is None:
                threshold = 1e-6
            else:
                threshold = np.abs(max_error)

            inlier_mask = sample_errors <= threshold
            inlier_count = np.sum(inlier_mask)


            # Best coeffs check
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inlier_mask
                best_coeffs = coeffs.copy()
            
            if inlier_count == population:
                break
        try:
            self.inlier_ratio = best_inlier_count / population if population > 0 else 0.0
            frac = self.inlier_ratio * 100 if population > 0 else 0.0
        except Exception:
            frac = 0.0

        if best_inliers is not None and best_inlier_count >= consensus:
            inlier_X = X[best_inliers]
            inlier_y = y[best_inliers]

            if len(inlier_X) >= self.poly_size:
                try:
                    ransac_coeffs = self._calc_coeffs(inlier_X, inlier_y)
                    if isinstance(ransac_coeffs, np.ndarray) and len(ransac_coeffs) == self.poly_size:
                        return ransac_coeffs
                except (np.linalg.LinAlgError, TypeError, ValueError):
                    pass # REEVALUTE `PASS`
                else:
                    return best_coeffs # Return best coeffs without refit 
                       
        if best_inliers is None or best_inlier_count < consensus:
            print(f"NO CONSENSUS! Best inlier's account for {frac}% of total population, but args required {self.min_inliers * 100}%. Falling back to full-data OLS.")
        # Leading Coefficient check: If leading coefficient is a negative value, fit all data
        if self.degree == 2 and best_coeffs is not None:
            if best_coeffs[-1] < 0:
                print(f"WARNING: Suspecious parabola a = {best_coeffs[-1]}")
                return self._calc_coeffs(X, y)

        # FAIL SAFE: Fit all data (ordinary least squares)
        try:
            print("FAIL SAFE FAILURE")
            last_resort = self._calc_coeffs(X, y)
            if isinstance(last_resort, np.ndarray) and len(last_resort) == self.poly_size:
                return last_resort
        except:
            pass # REEVALUTE `PASS`

    def predict(self, coeffs):
        # Generate 100 points in scaled space
        X_lin = np.linspace(0, 1, 100)

        # Estimate respective y-values in scaled space
        y_pred = self._poly_val(coeffs, X_lin)

        return X_lin, y_pred

    def _calc_coeffs(self, X, y):
        '''(X.T * X)**-1 * (X.T & y)'''
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        else:
            XT = X.T
            A = XT @ X

            if np.linalg.cond(A) > 1e10:
                raise np.linalg.LinAlgError("Matrix is ill-conditioned")
            b = XT @ y
            return np.linalg.solve(A, b).astype(np.ndarray)
        
    def _poly_val(self, coeffs, X):
        result = coeffs[-1]
        for coef in reversed(coeffs[:-1]):
            result = result * X + coef
        return result