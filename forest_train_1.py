import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import joblib


mnist = fetch_openml("mnist_784", version=1)

X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


for index in range(0, 59999):
    X_train.iloc[index].loc[X_train.iloc[index]<130] = 0
    X_train.iloc[index].loc[X_train.iloc[index]>100] = 253

for index in range(0, 1000):
    X_test.iloc[index].loc[X_test.iloc[index]<130] = 0
    X_test.iloc[index].loc[X_test.iloc[index]>100] = 253


from sklearn.ensemble import RandomForestClassifier

forest_clf_all = RandomForestClassifier(random_state=42)
forest_clf_all.fit(X_train, y_train)
joblib.dump(forest_clf_all, "forest_clf_all.pkl")

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

svc_clf_all = SVC(random_state=42)
svc_clf_all.fit(X_train, y_train)
joblib.dump(svc_clf_all, "svc_clf_all.pkl")