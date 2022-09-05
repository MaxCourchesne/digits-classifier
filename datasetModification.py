import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import joblib


mnist = fetch_openml("mnist_784", version=1)

X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


print(X_test[0])
from sklearn.ensemble import RandomForestClassifier

y_train_0 = (y_train=="0")

forest_clf_0 = RandomForestClassifier(random_state=42)
forest_clf_0.fit(X_train, y_train_0)
joblib.dump(forest_clf_0, "0_detector_forests.pkl")