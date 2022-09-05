import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import joblib

X_train = pd.read_csv("training_from drawing_X")
y_train = pd.read_csv("training_from drawing_y")
X_train = X_train.loc["data"]
y_train = y_train.loc["target"]


for index in range(0, 59999):
    X_train.iloc[index].loc[X_train.iloc[index]<130] = 0
    X_train.iloc[index].loc[X_train.iloc[index]>100] = 253

'''
    for index in range(0, 1000):
    X_test.iloc[index].loc[X_test.iloc[index]<130] = 0
    X_test.iloc[index].loc[X_test.iloc[index]>100] = 253
'''

from sklearn.ensemble import RandomForestClassifier

from sklearn.multiclass import OneVsOneClassifier

oneVsOne_clf_all = OneVsOneClassifier(estimator=RandomForestClassifier(random_state=42))
oneVsOne_clf_all.fit(X_train, y_train)
joblib.dump(oneVsOne_clf_all, "oneVsOne_clf_all.pkl")

