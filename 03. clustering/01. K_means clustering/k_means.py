from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('data/data.csv')
X = dataset.iloc[:,[1,3]].values

#using the elbow method
def elbow():
	wcss = []
	for i in range(1,12):
		kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
		kmeans.fit(X)
		wcss.append(kmeans.inertia_)
	 #plot the elbow
	plt.plot(range(1,12),wcss)
	plt.title('Elbow method')
	plt.xlabel('number of clusters')
	plt.ylabel('WCSS')
	plt.savefig('elbow.png')

#now we apply the k_means
kmeans = KMeans(n_clusters = 4, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 100, c='red',label = 'nothing')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 100, c='blue',label = 'nothing')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 100, c='green',label = 'nothing')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s = 100, c='cyan',label = 'nothing')

plt.savefig('clustering.png')