from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading the dataset
dataset = pd.read_csv('data/data.csv')
dataset = dataset.sort_values(by = ['interest rate (%)'])
X = (dataset['interest rate (%)']).values.reshape(-1,1)
y = np.array([float((rec.strip('$')).replace(',','')) for rec in dataset['Median home price']])

#fitting Decision tree
regressor = RandomForestRegressor(n_estimators = 400, random_state = 0)
regressor.fit(X,y)

#make a prediction
y_pred = regressor.predict(250000)

print y_pred

#visualize with high accuracy
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Random Forest Regression')
plt.savefig('Random_Forest_Regression.png')