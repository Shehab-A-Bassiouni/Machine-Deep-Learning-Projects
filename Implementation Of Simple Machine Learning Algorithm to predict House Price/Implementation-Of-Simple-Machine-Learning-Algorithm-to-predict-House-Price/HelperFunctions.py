import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class Helper:
    def RemoveStringColumns(data, columns):

        data.drop(data.columns[columns], inplace=True, axis=1)


    def GetDateFeature(data, column):

        data[column] = pd.to_datetime(data[column])
        data[column] = data[column].dt.strftime('%Y')
        data = data.to_numpy(dtype='int64')
        return data[:, 5]


    def LinearRegressionModel(feature, ground_truth):

        L = 0.00000001
        epochs = 75
        n = float(len(feature))
        m = 0
        c = 0
        for i in range(epochs):
            pred = m*feature + c
            dm = (-2 / n) * np.sum((ground_truth - pred) * feature)
            dc = (-2 / n) * np.sum(ground_truth - pred)
            m = m - (L * dm)
            c = c - (L * dc)
        pred = (m*feature) + c
        return c, m, pred



