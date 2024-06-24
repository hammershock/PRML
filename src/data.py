import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_adult():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
               "hours-per-week", "native-country", "income"]
    local_path = "../data/adult.csv"
    if os.path.exists(local_path):
        data = pd.read_csv(local_path, names=columns, sep=r'\s*,\s*', engine='python', na_values="?")
    else:
        data = pd.read_csv(url, names=columns, sep=r'\s*,\s*', engine='python', na_values="?")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        data.to_csv(local_path, index=False)

    data = data.dropna()

    categorical_columns = data.select_dtypes(include=['object']).columns
    data = data[categorical_columns]
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    X = data.drop("income", axis=1).to_numpy()
    y = data["income"].to_numpy()
    return X, y


def generate_ring(n_positive, n_negative, r=5.0, std=1):
    # Generate positive samples
    pos_samples = np.random.normal(loc=0, scale=1, size=(n_positive, 2))
    pos_labels = np.ones(n_positive)

    # Generate negative samples
    angles = np.random.uniform(0, 2 * np.pi, n_negative)
    radii = np.random.normal(loc=r, scale=std, size=n_negative)
    neg_samples = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    neg_labels = np.zeros(n_negative)

    # Combine positive and negative samples
    X = np.vstack((pos_samples, neg_samples))
    y = np.hstack((pos_labels, neg_labels))

    # Shuffle the dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y
