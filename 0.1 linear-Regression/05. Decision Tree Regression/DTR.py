from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading the dataset
dataset = pd.read_csv('data/data.csv')
dataset = dataset.sort_values(by = ['interest rate (%)'])
X = (dataset['interest rate (%)']).values.reshape(-1,1)
y = np.array([float((rec.strip('$')).replace(',','')) for rec in dataset['Median home price']])

#fitting Decision tree
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#make a prediction
y_pred = regressor.predict(250000)

print y_pred

#visualize with high accuracy
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Decision Tree Regression')
plt.savefig('Decison_Tree_Regression.png')