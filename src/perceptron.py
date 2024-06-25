class Perceptron:
    """The simplest linear classifier"""

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        N, dim = X.shape
        self.weights = np.zeros(dim)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                y_pred = np.sign(np.dot(x_i, self.weights) + self.bias)
                y_gt = y[idx]
                delta = y_gt - y_pred
                self.weights += delta * x_i * self.lr
                self.bias += delta * self.lr

    def predict(self, X):
        y_pred = np.sign(np.dot(X, self.weights) + self.bias)
        return y_pred


class DualPerceptron:
    """感知机的对偶形式"""
    def __init__(self):
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None

    def fit(self, X, y, max_iter=1000):
        N, dim = X.shape
        self.X, self.y = X, y

        self.alpha = np.zeros(N)  # dual version
        self.b = 0

        for _ in range(max_iter):
            for i in range(N):
                sum_alpha_y_k = np.sum(self.alpha * y * linear_kernel(X, X[i]))
                if y[i] * (sum_alpha_y_k + self.b) <= 0:  # mis-classified
                    self.alpha[i] += 1
                    self.b += y[i]

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            sum_alpha_y_k = np.sum(self.alpha * self.y * linear_kernel(self.X, X[i]))
            y_pred[i] = np.sign(sum_alpha_y_k + self.b)
        return y_pred


def linear_kernel(x, y):
    return np.dot(x, y)


if __name__ == '__main__':
    import numpy as np

    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([-1, -1, 1, 1])

    perceptron = DualPerceptron()
    perceptron.fit(X, y)

    preds = perceptron.predict(X)
    print(preds)

