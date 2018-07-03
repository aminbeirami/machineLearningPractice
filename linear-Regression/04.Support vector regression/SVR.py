from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading the dataset
dataset = pd.read_csv('data/data.csv')
dataset = dataset.sort_values(by = ['interest rate (%)'])
X = (dataset['interest rate (%)']).values.reshape(-1,1)
y = np.array([float((rec.strip('$')).replace(',','')) for rec in dataset['Median home price']])

#feature scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
print X.reshape(1,-1)
print y
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

#fitting regressor
regressor = SVR(kernel = 'rbf') #nonlinear regession
regressor.fit(X,y)

#predictinf a value
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[250000]]))))
print y_pred

#plotting
plt.scatter(X,y,color = 'red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.savefig('SVM.png')