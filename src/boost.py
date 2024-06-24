import warnings

import numpy as np

# from .data import generate_ring
from .base import BaseEstimator


class DecisionTreeStumpClassifier(BaseEstimator):
    """决策树桩分类器, X: 实数, y: 实数"""
    def __init__(self):
        self.boundary = None
        self.polarity = None
        self.feat_index = None
        self.err = None
        self._fitted = False

    def fit(self, X, y, w=None, **kwargs):
        for key in kwargs:
            warnings.warn(f"unexpected keyword argument {key}")
        N, dim = X.shape
        w = np.full(N, 1.0 / N) if w is None else w

        def _fit(feat, boundary, idx):
            nonlocal w, N, X, y
            pred = np.where(feat < boundary, -1, 1)
            err_mask = pred != y
            pol = -1 if np.sum(err_mask) > N / 2 else 1  # polarity
            err_mask = ~err_mask if pol == -1 else err_mask
            err = np.sum(w * err_mask)
            return err, pol, boundary, idx

        self.err, self.polarity, self.boundary, self.feat_index = (
            min(_fit(feat, bound, idx) for idx, feat in enumerate(X.T) for bound in np.unique(feat)))
        self._fitted = True
        return self

    def predict(self, X):
        assert self._fitted, "the estimator has not been fit yet"
        feat = X[:, self.feat_index]
        return np.where(feat < self.boundary, -1, 1) * self.polarity


class AdaBoostClassifier(BaseEstimator):
    def __init__(self, base_estimator=BaseEstimator):
        self.weights = None
        self.BaseEstimator = base_estimator
        self.estimators = []
        self.alphas = []

    def fit(self, X, y, weights=None, n_estimators=30):
        N, dim = X.shape
        assert set(y).issubset({-1, 1})
        self.weights = np.full(N, 1.0 / N) if weights is None else weights
        for _ in range(n_estimators):
            new_estimator = self.BaseEstimator().fit(X, y, self.weights)
            preds = new_estimator.predict(X)

            err = getattr(new_estimator, 'err', self.weights[preds != y].sum())
            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))

            self.weights *= np.exp(-alpha * y * preds)
            self.weights /= np.sum(self.weights)
            self.alphas.append(alpha)
            self.estimators.append(new_estimator)  # step forward
        return self

    def predict(self, X):
        alphas = np.array(self.alphas)
        preds = np.array([estimator.predict(X) for estimator in self.estimators])
        return np.sign(np.sum(alphas[:, None] * preds, axis=0))


# we use relative import, so it can not be run directly
# if __name__ == "__main__":
#     X, y = generate_ring(100, 200)
#     y[y == 0] = -1
#     clf = AdaBoostClassifier(base_estimator=DecisionTreeStumpClassifier).fit(X, y)
#     preds = clf.predict(X)
#     accuracy = np.sum(preds == y) / len(X)
#     print(f'{accuracy=}')
