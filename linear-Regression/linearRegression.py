from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Imports dataset
dataset = pd.read_csv('Data.csv')
x = (dataset['interest rate (%)']).values
y = [float((rec.strip('$')).replace(',','')) for rec in dataset['Median home price']]
# plt.scatter(x,y)
# plt.show()
#define the training set and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
#fitting Linear Regression to training set
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting y values based on X_test values
predict_y = regressor.predict(x_test)

#visualization
fig, ax = plt.subplots()
ax.scatter(x_train,y_train,color = 'red',label = 'available data')
ax.scatter(x_test,y_test,color = 'green',label = 'data to be predicted')
ax.scatter(x_test,predict_y,color = 'black',label = 'predicted data')
ax.plot(x_train,regressor.predict(x_train),color = 'blue')
ax.legend()
plt.xlabel('intrest rate')
plt.ylabel('salary (USD)')
plt.savefig('linearRegession.png')