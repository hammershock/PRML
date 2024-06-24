from abc import ABC, abstractmethod

import numpy as np


class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X, y, w=None, **kwargs):
        ...

    def predict(self, X) -> np.ndarray:
        ...
