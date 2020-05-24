"""
**Importing Libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""**Importing Datsets**"""

dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,3:].values
print(X)

"""**Elbow Method to determine no. of cluster in k means**"""

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
  km = KMeans(n_clusters=i,init="k-means++",random_state=0)
  km.fit(X)
  wcss.append(km.inertia_)
plt.plot(range(1,11),wcss,color="red")
plt.title("Elbow Method in kmeans to identify the no. of clusters")
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.show()

"""**Dendogram Method in Hierarchical Clustering to identify the number of clusters**"""

import scipy.cluster.hierarchy as sch
den = sch.dendrogram(sch.linkage(X,method="ward"))
plt.title("Dendrogram")
plt.xlabel("Points")
plt.ylabel("Euclidean Distance")
plt.show()

"""**Building kmeans Algo**"""

from sklearn.cluster import KMeans
kmf = KMeans(n_clusters=5,random_state=0,init="k-means++")
kmf.fit(X)

"""**Predicting and making our new Dependent Variable**"""

Ykmf = kmf.predict(X)

print(Ykmf)

"""**Building Hierarchical Model**"""

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
ac.fit(X)

yac = ac.fit_predict(X)

print(yac)

"""**Visualing by Graph - KMeans**"""

plt.scatter(X[Ykmf==0,0],X[Ykmf==0,1],label="Cluster-0",s=100,c="red")
plt.scatter(X[Ykmf==1,0],X[Ykmf==1,1],label="Cluster-1",s=100,c="green")
plt.scatter(X[Ykmf==2,0],X[Ykmf==2,1],label="Cluster-2",s=100,c="blue")
plt.scatter(X[Ykmf==3,0],X[Ykmf==3,1],label="Cluster-3",s=100,c="yellow")
plt.scatter(X[Ykmf==4,0],X[Ykmf==4,1],label="Cluster-4",s=100,c="pink")
plt.scatter(kmf.cluster_centers_[:,0],kmf.cluster_centers_[:,1],label="Centroid",s=200,c="cyan")
plt.title("Graph by Kmeans")
plt.xlabel("First Column of our Dataset")
plt.ylabel("Second Column of our Dataset")
plt.legend()
plt.show()

print(kmf.cluster_centers_)

"""**Visualing by Graph - Heirarichal**"""

plt.scatter(X[yac==0,0],X[yac==0,1],label="Cluster-0",s=100,c="red")
plt.scatter(X[yac==1,0],X[yac==1,1],label="Cluster-1",s=100,c="green")
plt.scatter(X[yac==2,0],X[yac==2,1],label="Cluster-2",s=100,c="blue")
plt.scatter(X[yac==3,0],X[yac==3,1],label="Cluster-3",s=100,c="yellow")
plt.scatter(X[yac==4,0],X[yac==4,1],label="Cluster-4",s=100,c="pink")
# plt.scatter(ac.cluster_centers_[:,0],ac.cluster_centers_[:,1],label="Centroid",s=200,c="cyan")
plt.title("Graph By Heirarichal")
plt.legend()
plt.xlabel("First Column of our Dataset")
plt.ylabel("Second Column of our Dataset")
plt.show()
