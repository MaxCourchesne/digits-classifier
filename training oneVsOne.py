import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import joblib


mnist = fetch_openml("mnist_784", version=1)

X, y = mnist["data"], mnist["target"]
X_train, y_train = X[:69999], y[:69999]


for index in range(0, 69999):
    X_train.iloc[index].loc[X_train.iloc[index]<130] = 0
    X_train.iloc[index].loc[X_train.iloc[index]>100] = 253



from sklearn.ensemble import RandomForestClassifier

from sklearn.multiclass import OneVsOneClassifier

oneVsOne_clf_all = OneVsOneClassifier(estimator=RandomForestClassifier(random_state=42))
oneVsOne_clf_all.fit(X_train, y_train)
joblib.dump(oneVsOne_clf_all, "oneVsOne_clf_all.pkl")

