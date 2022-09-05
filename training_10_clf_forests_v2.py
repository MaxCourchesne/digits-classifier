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


y_train_0 = (y_train=="0")
y_test_0 = (y_test=="0")

y_train_1 = (y_train=="1")
y_test_1 = (y_test=="1")

y_train_2 = (y_train=="2")
y_test_2 = (y_test=="2")

y_train_3 = (y_train=="3")
y_test_3 = (y_test=="3")

y_train_4 = (y_train=="4")
y_test_4 = (y_test=="4")

y_train_5 = (y_train=="5")
y_test_5 = (y_test=="5")

y_train_6 = (y_train=="6")
y_test_6 = (y_test=="6")

y_train_7 = (y_train=="7")
y_test_7 = (y_test=="7")

y_train_8 = (y_train=="8")
y_test_8 = (y_test=="8")

y_train_9 = (y_train=="9")
y_test_9 = (y_test=="9")

from sklearn.ensemble import RandomForestClassifier

forest_clf_0 = RandomForestClassifier(random_state=42)
forest_clf_0.fit(X_train, y_train_0)
joblib.dump(forest_clf_0, "0_detector_forests_v2.pkl")

forest_clf_1 = RandomForestClassifier(random_state=42)
forest_clf_1.fit(X_train, y_train_1)
joblib.dump(forest_clf_1, "1_detector_forests_v2.pkl")

forest_clf_2 = RandomForestClassifier(random_state=42)
forest_clf_2.fit(X_train, y_train_2)
joblib.dump(forest_clf_2, "2_detector_forests_v2.pkl")

forest_clf_3 = RandomForestClassifier(random_state=42)
forest_clf_3.fit(X_train, y_train_3)
joblib.dump(forest_clf_3, "3_detector_forests_v2.pkl")

forest_clf_4 = RandomForestClassifier(random_state=42)
forest_clf_4.fit(X_train, y_train_4)
joblib.dump(forest_clf_4, "4_detector_forests_v2.pkl")

forest_clf_5 = RandomForestClassifier(random_state=42)
forest_clf_5.fit(X_train, y_train_5)
joblib.dump(forest_clf_5, "5_detector_forests_v2.pkl")

forest_clf_6 = RandomForestClassifier(random_state=42)
forest_clf_6.fit(X_train, y_train_6)
joblib.dump(forest_clf_6, "6_detector_forests_v2.pkl")

forest_clf_7 = RandomForestClassifier(random_state=42)
forest_clf_7.fit(X_train, y_train_7)
joblib.dump(forest_clf_7, "7_detector_forests_v2.pkl")

forest_clf_8 = RandomForestClassifier(random_state=42)
forest_clf_8.fit(X_train, y_train_8)
joblib.dump(forest_clf_8, "8_detector_forests_v2.pkl")

forest_clf_9 = RandomForestClassifier(random_state=42)
forest_clf_9.fit(X_train, y_train_9)
joblib.dump(forest_clf_9, "9_detector_forests_v2.pkl")
