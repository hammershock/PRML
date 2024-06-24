from collections import Counter, defaultdict
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class NaiveBayesClassifier:
    """X: 实数, y: 类别"""
    def __init__(self):
        self.priori_y = {}
        self.priori_x = defaultdict(dict)
        self._fitted = False

    def fit(self, X, y, w=None):
        for y_val, cnt in Counter(y).items():
            self.priori_y[y_val] = cnt / len(y)

        y_given_x_counts = defaultdict(lambda: defaultdict(list))
        for idx, col in enumerate(X.T):
            y_given_x_accumulator: Dict[Any, list] = y_given_x_counts[idx]

            for x_val, y_val in zip(col, y):
                y_given_x_accumulator[x_val].append(y_val)

            # P_y_given_X: Dict[Any, Dict[Any, float]] = {}
            for x_val, lst in y_given_x_accumulator.items():
                probs = {key: value / len(lst) for key, value in Counter(lst).items()}
                self.priori_x[idx][x_val] = probs

        self._fitted = True
        return self

    def predict_one(self, X):
        assert self._fitted, f"NaiveBayesClassifier not fitted, use fit(X, y) first"
        prob = self.priori_y.copy()
        for idx, x_val in enumerate(X):
            for y_val in prob:
                p = self.priori_x[idx].get(x_val, {}).get(y_val, 0)
                prob[y_val] *= p
        return max(prob, key=lambda k: prob[k])

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


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

    cls = NaiveBayesClassifier()
    cls.fit(X, y)

    most_y = Counter(y).most_common()[0][1]
    print(most_y / len(y))  # 0.7510775147536636
    print(np.sum(cls.predict(X) == y) / len(y))  # 0.7510775147536636
