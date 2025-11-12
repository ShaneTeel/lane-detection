import numpy as np
from lane_detection.models.ols_regression import OLSRegression

class RANSACRegression():
    '''Test'''

    def __init__(self, degree:int = 2, n_iter:int=50, min_inliers:float = 0.8, max_error:int = 10):

        self.ols = OLSRegression(degree)
        self.poly_size = self.ols.poly_size
        self.sample_size = self.poly_size + 2
        self.n_iter = n_iter
        self.min_inliers = min_inliers
        self.max_error = max_error
        self.inlier_ratio = None
        self.name = "RANSAC Regression"

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
        if population < self.ols.poly_size:
            return self.ols.fit(X, y)

        # Use the minimal sample size necessary for the model (poly_size)
        sample_size = min(max(self.sample_size, 1), population)
        
        for _ in range(self.n_iter):
            sample_X, sample_y = self._rand_sampling(X, y, population, sample_size)

            # Fit polynomial to samples
            try:
                coeffs = self.ols.fit(sample_X, sample_y)
    
                if not isinstance(coeffs, np.ndarray) or len(coeffs) != self.ols.poly_size:
                    print(f"WARNING: calcualted coeffs not of correct type ({np.ndarray}) or correct size ({self.ols.poly_size})")
                    continue

            except (np.linalg.LinAlgError, TypeError, ValueError) as e:
                print(f"WARNING: polyfit error - {e}")
                continue
            
            # Evaluate sample fit on all points (original scaled X)
            inlier_count, inlier_mask = self._evaluate_fit(coeffs, X, y, max_error)

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

            if len(inlier_X) >= self.ols.poly_size:
                try:
                    ransac_coeffs = self.ols.fit(inlier_X, inlier_y)
                    if isinstance(ransac_coeffs, np.ndarray) and len(ransac_coeffs) == self.ols.poly_size:
                        return ransac_coeffs
                except (np.linalg.LinAlgError, TypeError, ValueError):
                    pass # REEVALUTE `PASS`
                else:
                    return best_coeffs # Return best coeffs without refit 
                       
        if best_inliers is None or best_inlier_count < consensus:
            print(f"NO CONSENSUS! Best inlier's account for {frac}% of total population, but args required {self.min_inliers * 100}%. Falling back to full-data OLS.")
        # Leading Coefficient check: If leading coefficient is a negative value, fit all data
        if self.ols.degree == 2 and best_coeffs is not None:
            if best_coeffs[-1] < 0:
                print(f"WARNING: Suspecious parabola a = {best_coeffs[-1]}")
                return self.ols.fit(X, y)

        # FAIL SAFE: Fit all data (ordinary least squares)
        try:
            print("FAIL SAFE FAILURE")
            last_resort = self.ols.fit(X, y)
            if isinstance(last_resort, np.ndarray) and len(last_resort) == self.ols.poly_size:
                return last_resort
        except:
            pass # REEVALUTE `PASS`

    def predict(self, coeffs):
        return self.ols.predict(coeffs)
    
    def _evaluate_fit(self, coeffs, X, y, max_error):
        y_pred = self.ols._poly_val(coeffs, X)

        # Use absolute error
        sample_errors = np.abs(y - y_pred)

        # Count inliers (points close to fit)
        # Ensure max_error is a scalar; if None, treat as very small to avoid accepting everything
        if max_error is None:
            threshold = 1e-6
        else:
            threshold = max_error

        inlier_mask = sample_errors <= threshold
        inlier_count = np.sum(inlier_mask)
        return inlier_count, inlier_mask
    
    def _rand_sampling(self, X, y, population, sample_size):
        # Random sampling
        sample_idx = np.random.choice(population, size=sample_size, replace=False)
        sample_X = X[sample_idx]
        sample_y = y[sample_idx]
        return sample_X, sample_y