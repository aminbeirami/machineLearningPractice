from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#reading the dataset
dataset = pd.read_csv('Data/Data2.csv')
dataset = dataset.sort_values(by = ['interest rate (%)'])
X = (dataset['interest rate (%)']).values.reshape(-1,1)
y = [float((rec.strip('$')).replace(',','')) for rec in dataset['Median home price']]

# fitting polynomial regression to the dataset
poly_regression = PolynomialFeatures(degree = 6) # this adds another column(s) to the dataset
X_polynomial = poly_regression.fit_transform(X) # we need to fit and transform newly made dataset into X_pol variable
poly_regression.fit(X_polynomial,y)
regressor = LinearRegression()
regressor.fit(X_polynomial,y)

plt.scatter(X,y,color = 'red')
plt.plot(X, regressor.predict(poly_regression.fit_transform(X)),color = 'blue')
plt.savefig('plynomial_linear_regression.png')