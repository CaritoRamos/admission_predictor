# custom_transformers.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def softM(x, LAMBDA=2):
    PI = np.pi
    AVG = np.mean(x)
    SD = np.std(x)
    vt = (x - AVG) / (LAMBDA * (SD / (2 * np.pi)))
    return 1 / (1 + np.exp(-vt))

class softMax(BaseEstimator, TransformerMixin):
    def __init__(self, LAMBDA=2):
        self.LAMBDA = LAMBDA

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "apply"):
            return X.apply(lambda col: softM(col.values, LAMBDA=self.LAMBDA), axis=0)
        else:
            return softM(X, LAMBDA=self.LAMBDA)