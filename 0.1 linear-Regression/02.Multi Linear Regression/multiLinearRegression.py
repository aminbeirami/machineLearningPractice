from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#the dataset is about different CPU performance
#we are intrested to create a linear model out of training data
#our depenant variable is the last column of the csv file (Estimated Relative Performance)
dataset = pd.read_csv('Data/data.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#transforming categorical data.
#if there are no categorical variables, skip this step

# labelencoder_X = LabelEncoder()
# X[:,0] = labelencoder_X.fit_transform(X[:,0])
# X[:,1] = labelencoder_X.fit_transform(X[:,1])
# onehotencoder = OneHotEncoder(categorical_features = [0,1])
# X = onehotencoder.fit_transform(X).toarray()

#To run away from dummy variable trap we delete one of the dummy variables
X = X[:,1:]

#splitting training and testing data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state= 0)

#fitting multiple linear regression to training set

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_prediction = regressor.predict(X_test)

#adding column of ones to the beginning of the dataset. this is because the form is b0+b1X1+...+bnXn. withoud adding ones we eliminate the b0
X = np.append(arr = np.ones((60,1)).astype(int), values = X, axis = 1)
# #Backward elimination to remove less significant variables
#check the statistics, look for P value. set a significance level and if P value of a variable was above the significance value, then remove it
#significance level I chose was P< 0.05
X_opt = X[:,[0,1,5,7,8,11,12]] #start with number of independant variables and then remove step by step as P value is checked
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #ordinary least squares
print regressor_OLS.summary()

# the remained X_opt are the independant variables that have significant impact on the dependant variable.


