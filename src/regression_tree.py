import numpy as np


class RegressionTreeNode(object):
    def __init__(self, X, y, w):
        self.alpha_ = None
        self.Nt_ = None
        self.C_ = None

        def loss_fn(j, col, s):
            left = col < s
            lp, rp = y[left], y[~left]
            lw, rw = w[left], w[~left]
            l_var = np.sum((lp - np.mean(lp)) ** 2 * lw) if len(lp) else 0.0
            r_var = np.sum((rp - np.mean(rp)) ** 2 * rw) if len(rp) else 0.0
            return l_var + r_var, j, s, left

        self.value = np.mean(y)
        # find sub-optimal feature-dim and split point
        lsq = min((loss_fn(j, col, s) for j, col in enumerate(X.T) for s in col), key=lambda x: x[0])
        self.c, self.j, self.s, left = lsq

        if np.sum(left) * np.sum(~left) == 0:
            self.children = None
            return

        self.children = RegressionTreeNode(X[left], y[left], w[left]), RegressionTreeNode(X[~left], y[~left], w[~left])

    def predict(self, X):
        if self.is_leaf:
            return np.full(X.shape[0], self.value)

        left = X[:, self.j] < self.s
        y_pred = np.empty(X.shape[0])
        y_pred[left] = self.children[0].predict(X[left])
        y_pred[~left] = self.children[1].predict(X[~left])
        return y_pred

    def update_(self):
        if self.is_leaf:
            self.C_ = self.c
            self.Nt_ = 1
            return
        self.children[0].update_()
        self.children[1].update_()
        self.C_ = self.children[0].C_ + self.children[1].C_
        self.Nt_ = self.children[0].Nt_ + self.children[1].Nt_
        self.alpha_ = (self.c - self.C_) / (self.Nt_ - 1)

    @property
    def is_leaf(self):
        return self.children is None

    def prune_(self, alpha):
        if self.is_leaf:
            return
        self.children[0].prune_(alpha)
        self.children[1].prune_(alpha)

        if self.children[0].is_leaf and self.children[1].is_leaf:
            if self.alpha_ < alpha:
                self.children = None


class LSqRegressionTree(object):
    """最小二乘回归树, 二叉回归树, X: 实数, y: 实数"""

    def __init__(self):
        self.root = None
        self._fitted = False

    def fit(self, X, y, weights=None):
        N, dim = X.shape
        weights = np.full((N, 1), 1 / N) if weights is None else weights
        self.root = RegressionTreeNode(X, y, weights)
        self._fitted = True
        return self

    def predict(self, X):
        assert self._fitted, "this tree has not been fit yet"
        return self.root.predict(X)

    def prune(self, alpha=1.0):
        """最小化带有正则项alpha的误差函数，每个叶子结点带有alpha的惩罚"""
        self.root.update_()
        self.root.prune_(alpha)


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X, y = data.data, data.target
    tree = LSqRegressionTree().fit(X, y)
    print(np.std(y - tree.predict(X)))
    tree.prune(0.1)
    print(np.std(y - tree.predict(X)))
