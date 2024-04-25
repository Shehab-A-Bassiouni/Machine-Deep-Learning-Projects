import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from HelperFunctions import Helper

## TODO: [1] Read the Data.csv ##
## Start Code ## (≈1 line)
data = pd.read_csv('Houses.csv')
## End Code ##

## TODO: [2] Clean the DataFrame Features ##
## TODO: [2.1] Remove Columns with String Values from your DataFrame ##
## TODO: [2.2] Change the `date_added` feature to datetime format ##
## TODO: Use RemoveStringColumns() & GetDateFeature() ##
## Start Code ## (≈3 lines)
Helper.RemoveStringColumns(data, [0, 2, 3, 7])
year = Helper.GetDateFeature(data, 'date_added')
data = data.to_numpy(dtype='float64')
## End Code ##

## TODO: [3] Build the Simple Linear Regression Models ##
## TODO: Use LinearRegressionModel() ##
## Start Code ##
c1, m1, pred1 = Helper.LinearRegressionModel(data[:, 1], data[:, 0])
c2, m2, pred2 = Helper.LinearRegressionModel(data[:, 2], data[:, 0])
c3, m3, pred3 = Helper.LinearRegressionModel(data[:, 3], data[:, 0])
c4, m4, pred4 = Helper.LinearRegressionModel(data[:, 4], data[:, 0])
c5, m5, pred5 = Helper.LinearRegressionModel(data[:, 5], data[:, 0])
c6, m6, pred6 = Helper.LinearRegressionModel(data[:, 6], data[:, 0])
## End Code ##

## TODO: [4] Plot your Learned Models against the Feature Values ##
## Start Code ##

plt.scatter(data[:, 1], data[:, 0])
plt.xlabel('latitude', fontsize = 20)
plt.ylabel('price', fontsize = 20)
plt.plot(data[:, 1], pred1, color='red', linewidth = 3)
plt.show()

plt.scatter(data[:, 2], data[:, 0])
plt.xlabel('longitude', fontsize = 20)
plt.ylabel('price', fontsize = 20)
plt.plot(data[:, 2], pred2, color='red', linewidth = 3)
plt.show()

plt.scatter(data[:, 3], data[:, 0])
plt.xlabel('baths', fontsize = 20)
plt.ylabel('price', fontsize = 20)
plt.plot(data[:, 3], pred3, color='red', linewidth = 3)
plt.show()

plt.scatter(data[:, 4], data[:, 0])
plt.xlabel('bedrooms', fontsize = 20)
plt.ylabel('price', fontsize = 20)
plt.plot(data[:, 4], pred4, color='red', linewidth = 3)
plt.show()

plt.scatter(data[:, 5], data[:, 0])
plt.xlabel('date_added', fontsize = 20)
plt.ylabel('price', fontsize = 20)
plt.plot(data[:, 5], pred5, color='red', linewidth = 3)
plt.show()

plt.scatter(data[:, 6], data[:, 0])
plt.xlabel('Total_Area', fontsize = 20)
plt.ylabel('price', fontsize = 20)
plt.plot(data[:, 6], pred6, color='red', linewidth = 3)
plt.show()

print('Mean Square Error for latitude', mean_squared_error(data[:, 1], pred1))
print('Mean Square Error longitude', mean_squared_error(data[:, 2], pred2))
print('Mean Square Error baths', mean_squared_error(data[:, 3], pred3))
print('Mean Square Error bedrooms', mean_squared_error(data[:, 4], pred4))
print('Mean Square Error date_added', mean_squared_error(data[:, 5], pred5))
print('Mean Square Error Total_Area', mean_squared_error(data[:, 6], pred6))
## End Code ##


