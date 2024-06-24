from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .base import BaseEstimator


def ent(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p)) if len(p) > 0 else 0.0


def gini(p):
    return 1.0 - np.sum(p ** 2)


class ClassificationTreeNode:
    def __init__(self, X, y, Dxs, Dy, indices: set, gain_threshold=0.0, score_algorithm='gini'):
        """C4.5 algorithm"""

        def freq(arr, attr):
            if len(arr) == 0:
                return np.zeros(0)
            return np.array([np.sum(arr == i) / len(arr) for i in attr])

        self.value, cnt = Counter(y).most_common(1)[0]
        self.children = {}
        self.scorer = gini if score_algorithm == 'gini' else ent

        if cnt == len(y):  # 所有y都是同一个标签，没有必要继续分下去了
            self.idx = None
            return
        elif len(indices) == 0:  # 特征集为空，所有的区分特征都被用掉了，还是没办法完全区分开，就没有必要再分下去了
            self.idx = None
            return

        H_D = self.scorer(freq(y, Dy))  # 经验熵，越低分类不确定度越小

        def score(idx):
            nonlocal H_D
            Dx = Dxs[idx]
            col = X[:, idx]
            p_attrs = freq(col, Dx)
            H_DA = 0.0  # 条件熵
            for p_val, val in zip(p_attrs, Dx):
                p = freq(y[col == val], Dy)
                H_DA += p_val * self.scorer(p)

            HA_d = self.scorer(p_attrs)  # 属性A的熵，越小越好
            GainRatio = (H_D - H_DA) / (HA_d + 1e-10)  # 信息增益比
            return GainRatio, idx

        # 寻找信息增益比最大的属性，为最优切分属性
        gain_ratio, self.idx = max(score(idx) for idx in indices)
        if gain_ratio < gain_threshold:
            self.idx = None
            return

        # 以此属性的不同取值切分数据集
        col = X[:, self.idx]
        values = np.unique(col)
        for value in values:
            mask = col == value
            new_indices = indices.copy() - {self.idx}
            # 对于D中A=a的子集
            self.children[value] = ClassificationTreeNode(X[mask], y[mask], Dxs, Dy, new_indices, gain_threshold, score_algorithm)

    def predict_single(self, x):
        if self.idx is None:  # 叶子结点，没有特征作为切分
            return self.value
        child = self.children.get(x[self.idx])
        if child:
            return child.predict_single(x)  # 如果以x[self.idx]为切分的子结点存在
        else:
            return self.value  # 否则直接返回当前结点的估计值


class DecisionTreeClassifier(BaseEstimator):
    """X: 类别(int), y: 类别(int)"""
    def __init__(self):
        self.root = None
        self._fitted = False
        self.Dxs = None
        self.Dy = None

    def fit(self, X, y, w=None, gain_threshold=0.01, score_algorithm='gini', **kwargs):
        self.Dxs = [np.unique(col) for col in X.T]
        self.Dy = np.unique(y)
        self.root = ClassificationTreeNode(X, y, self.Dxs, self.Dy, set(range(X.shape[1])), gain_threshold, score_algorithm)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise Exception("The model is not fitted yet.")
        return np.array([self.root.predict_single(x) for x in X])


if __name__ == '__main__':
    data = pd.read_csv("./data/categorical_features.csv", sep=r'\s*,\s*', engine='python', na_values="?")
    # data.to_csv('./data/data.csv', index=False)
    data = data.dropna()

    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    X = data.drop("income", axis=1).to_numpy()
    y = data["income"].to_numpy()

    tree = DecisionTreeClassifier().fit(X, y, score_algorithm='gini')
    pred = tree.predict(X)
    print(np.sum(pred == y) / len(pred))
