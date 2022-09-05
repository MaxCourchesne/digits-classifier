
import pygame
import numpy as np
import random
import pandas as pd
import pickle as pkl
import sklearn
from sklearn.datasets import fetch_openml
import joblib
import time
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist["data"], mnist["target"]
df_X = pd.DataFrame(X)
df_y = pd.DataFrame(y)
df_X.to_csv("training_from drawing_X")
df_y.to_csv("training_from drawing_y")
print(f"df_X = {df_X}")
print(f"df_y = {df_y}")

print(f"would be appended at {len(df_y)}")