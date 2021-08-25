import pywt
import numpy as np
from sklearn.decomposition import PCA
from calculate_statistical import get_statistic_features
from numpy import genfromtxt
import pandas as pd

X = genfromtxt(r"C:\data\temperature_timeseries.txt", delimiter=" ")


pca = PCA(n_components='mle', whiten=False)
X = pca.fit_transform(X)
newX = np.array_split(X, 100)

# print(len(newX))
list_features = []
for i in range(0, len(newX)):
    for j in range(0, len(newX[0][0])):
        features = []
        list_coefficient = pywt.wavedec(X[:, j], 'sym5')
        for coefficient in list_coefficient:
            features += get_statistic_features(coefficient)
        list_features.append(features)
# dl = pd.DataFrame(list_features)
# print(dl)
#tach 70