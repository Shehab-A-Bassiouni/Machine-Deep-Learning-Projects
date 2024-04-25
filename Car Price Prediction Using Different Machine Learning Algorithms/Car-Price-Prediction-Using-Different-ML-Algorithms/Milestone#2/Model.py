import warnings
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

train_data = pd.read_csv("datasets/train_preprocessed.csv")
test_data = pd.read_csv("datasets/test_preprocessed.csv")

X = train_data.loc[:, train_data.columns != "Price Category"]
X = X.loc[:, X.columns != "car_id"]
Y = train_data["Price Category"]

tester = test_data.loc[:, test_data.columns != "car_id"]

random.seed(10)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=random.randint(0, 50))

pca = PCA(0.99)
pca.fit(X)
xtrain = pca.transform(xtrain)
xtest = pca.transform(xtest)
tester = pca.transform(tester)

# ---------------------------------------------SVC----------------------------------------
# reducing C value lower than 1 reduce the accuracy
# kernal = linear | c=0.001  -----> accuracy = 0.8444839405865177
# kernal = linear | c=1  -----> accuracy = 0.877872286403453
# kernal = poly | degree = 7  -----> accuracy = 0.8418179509965723
# kernal = poly | degree = 3  -----> accuracy = 0.8880284372222927
# kernal = poly | degree = 2  -----> accuracy = 0.8680969912403199
# Kernal rbf | gamma = 0.8 -----> accuracy = 0.8708899327155009
# Kernal rbf | gamma = scale -----> accuracy = 0.8922178494350641 (Best)

# model = SVC(kernel='rbf', gamma='scale')
# model.fit(xtrain, ytrain)
# pred = model.predict(xtest)

# ---------------------------------RandomForestCLF---------------------------------------------------
# n_estimators = 100 ---> accuracy = 0.8815538910752825
# n_estimators = 500 ---> accuracy = 0.8821886505014599

# model = RandomForestClassifier(n_estimators=500)
# model.fit(xtrain,ytrain)
# pred=model.predict(xtest)

# ---------------------------------KNeighborsCLF---------------------------------------------------
# n_neighbors = 5 | algorithm = auto ------> accuracy =  0.8416909991113368
# n_neighbors = 20  | algorithm = auto ------> accuracy = 0.8447378443569887
# n_neighbors = 5  | algorithm = kd_tree ------> accuracy = 0.8416909991113368
# n_neighbors = 5  | algorithm = brute ------> accuracy = 0.8416909991113368

model = KNeighborsClassifier( n_neighbors = 20)
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
# --------------------------------------------------------------------------------------------------


print("accuracy = ", np.mean(pred == ytest))


plt.figure(figsize=(10, 10))
plt.scatter(ytest, pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(pred), max(ytest))
p2 = min(min(pred), min(ytest))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()


test_data["Price Category"] = model.predict(tester)
test_data.drop(test_data.columns.difference(['car_id', 'Price Category']), 1, inplace=True)

pricenum_to_cat = {"Price Category": {0: "cheap", 1: "moderate", 2: "expensive", 3: "very expensive"}}
test_data = test_data.replace(pricenum_to_cat)
test_data.to_csv("datasets/test_result_KNeighborsClassifier.csv", index=False)
