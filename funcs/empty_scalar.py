# a class that acts like standerd scalar from skikit but does nothing just passes the data through the same 
# with no transformations
import numpy as np
import pandas as pd 

class EmptyScaler:
    """
    A scaler that does nothing - passes data through unchanged.
    Compatible with scikit-learn's StandardScaler API.
    """
    
    def __init__(self):
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        """Fit the scaler (does nothing but marks as fitted)."""
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """Transform the data (returns unchanged)."""
        return np.asarray(X)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step (returns unchanged)."""
        self.fit(X, y)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform (returns unchanged)."""
        return np.asarray(X)
    
    def get_params(self, deep=True):
        """Get parameters (empty dict)."""
        return {}
    
    def set_params(self, **params):
        """Set parameters (does nothing)."""
        return self