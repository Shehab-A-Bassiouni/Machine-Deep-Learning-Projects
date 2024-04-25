import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('assignment2_bmd.csv')
data.dropna(axis=0,how='any',inplace=True)
X=data.iloc[ : , 0:7]
Y=data["bmd"]
feature = data.iloc[:,:]

col = ['sex' , 'fracture' ,'medication']
for i in col:
    encode= LabelEncoder()
    encode.fit(list(X[i].values))
    X[i]=encode.transform(list(X[i].values))

corr=feature.corr()
top=corr.index[abs(corr['bmd']) > 0.4]

plt.subplots(figsize=(12, 8))
top_corr = feature[top].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top = top.delete(-1)
X = X[top]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True,random_state=10)

poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)

poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_train_predicted = poly_model.predict(X_train_poly)
prediction = poly_model.predict(poly_features.fit_transform(X_test))

print('Co-efficient of linear regression',poly_model.coef_)
print('Intercept of linear regression model',poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

bdm_original=np.asarray(y_test)[0]
predicted_bdm=prediction[0]
print('original bdm for first patient in test set is : ' + str(bdm_original))
print('Predicted bdm for first patient is : ' + str(predicted_bdm))

