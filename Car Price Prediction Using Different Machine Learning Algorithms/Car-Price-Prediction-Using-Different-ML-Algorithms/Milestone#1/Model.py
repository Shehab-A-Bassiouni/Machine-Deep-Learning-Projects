import warnings
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

train_data = pd.read_csv("datasets/train_preprocessed.csv")
test_data = pd.read_csv("datasets/test_preprocessed.csv")

X = train_data.loc[:, train_data.columns != "price(USD)"]
X = X.loc[:, X.columns != "car_id"]
Y = train_data["price(USD)"]

tester = test_data.loc[:, test_data.columns != "car_id"]


random.seed(10)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=random.randint(0, 50))

pca = PCA(0.99)
pca.fit(X)
xtrain = pca.transform(xtrain)
xtest = pca.transform(xtest)
tester = pca.transform(tester)

# 3022.3072687408157
pr = PoissonRegressor(max_iter=1000)
pr.fit(xtrain, ytrain)
predict = pr.predict(xtest)

# 2653.9927372962597
# pr = RandomForestRegressor()
# pr.fit(xtrain, ytrain)
# predict = pr.predict(xtest)

mse = mean_squared_error(ytest, predict)
print("RMSE Model 1 ", mse ** (1 / 2.0))

plt.figure(figsize=(10, 10))
plt.scatter(ytest, predict, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(predict), max(ytest))
p2 = min(min(predict), min(ytest))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

test_data["price(USD)"] = pr.predict(tester)
test_data.drop(test_data.columns.difference(['car_id', 'price(USD)']), 1, inplace=True)
test_data.to_csv("datasets/test_result_PoissonRegressor.csv", index=False)
