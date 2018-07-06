from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('data/data.csv')
X = dataset.iloc[:,[1,3]].values

#plotting the dendrogram
def dendrogram_method():
	dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
	#plot the elbow
	plt.title('Dendrogram')
	plt.xlabel('nodes')
	plt.ylabel('Euclidean distance')
	plt.savefig('dendrogram.png')

#now we apply the hierarchical clustering
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s = 100, c='red',label = 'nothing')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s = 100, c='blue',label = 'nothing')

plt.savefig('Hierarchical.png')