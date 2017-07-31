import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import VarianceThreshold

from pandas.tools.plotting import parallel_coordinates, radviz
from mpl_toolkits.mplot3d import Axes3D

__author__ = "Jean Raltique - LibPhys"
__version__ = "1.0"
__license__ = "Universidade Nova de Lisboa - FCT - Physics Department"



'''
------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------Clustering script------------------------------------------------------
This script focuses in executing the clustering methods presented by the
sklearn platform.
In this, the methods Kmeans, DBSCAN, Affinity propagation, etc...can be called
in order to execute a classification by clusters.
Typically, the matrix of features will be pre-processed in order to optimize the
clustering method.
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
'''

def MultiDimensionalClusteringKmeans(Xmatrix, n_clusters=2):
	"""
	Perform a Multidimensional clustering with Kmeans method.
		The method accepts :
		- The feature matrix - Xmatrix
		- The time vector - time
		- The original Signal - xdata
		- The number of Clusters - n_clusters (which is 2 by default)
		- Which axes the plot may belong - ax (which is None by default)
		- The selection of plot or not - show (which is None by default)

	The method starts to normalize the matrix by a StandardScaler which
	optimize the computational effort.
	Then, the Kmeans object is created, and can be executed in order to
	classify the data.

	Finally, a PCA is performed in order to select the two features with most
	variance and that can give a better representation of the signal.
	"""

	seed = np.random.seed(0)
	colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
	colors = np.hstack([colors] * 20)

	#normalize dataset for easier parameter selection
	X = StandardScaler().fit_transform(Xmatrix)

	pca = PCA(n_components='mle')

	X_pca = pca.fit_transform(X)

	varExpl = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
	print("Variance Explained: ", str(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)))
	params = pca.get_params(deep=True)


	#select number of features to use on clustering (enough to exlpain 90% of the variance)
	if(varExpl[-1] > 95):
		n_features = np.where(varExpl>95)[0][0]
	else:
		n_features = len(varExpl) - 1


	#algorithm KMeans
	kmeans = KMeans(n_clusters=n_clusters, random_state=seed)

	#Apply algorithm
	kmeans.fit(X_pca[:, 0:n_features+1])

	y_pred = kmeans.labels_.astype(np.int)
	centers = kmeans.cluster_centers_
	center_colors = colors[:len(centers)]

	# # Step size of the mesh. Decrease to increase the quality of the VQ.
	# h = .02  # point in the mesh [x_min, m_max]x[y_min, y_max].
	#
	# # Plot the decision boundary. For that, we will assign a color to each
	# x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
	# y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
	# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	#
	# # Obtain labels for each point in mesh. Use last trained model.
	# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
	#
	# # Put the result into a color plot
	# Z = Z.reshape(xx.shape)


	return X, y_pred, X_pca[:,0:2], params

def MultiDimensionalClusteringAGG(Xmatrix, n_clusters=2, Linkage = 'ward', Affinity = 'euclidean'):

	# normalize dataset for easier parameter selection
	X = StandardScaler().fit_transform(Xmatrix)



	pca = PCA(n_components='mle')
	#kpca = KernelPCA(n_components=2, kernel='sigmoid')
	# X_kpca  = kpca.fit_transform(X)

	X_pca = pca.fit_transform(X)

	varExpl = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)
	print("Variance Explained: ", str(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)))
	params = pca.get_params(deep=True)


	#select number of features to use on clustering (enough to exlpain 90% of the variance)
	if(varExpl[-1] > 95):
		n_features = np.where(varExpl>95)[0][0]
	else:
		n_features = len(varExpl) - 1

	# connectivity matrix for structured Ward
	connectivity = kneighbors_graph(X_pca[:, 0:n_features+1], mode='distance', n_neighbors=20, include_self=False)
	# make connectivity symmetric
	connectivity = 0.5 * (connectivity + connectivity.T)
	print(connectivity)

	seed = np.random.seed(0)

	if(Linkage is 'ward'):
		ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity, affinity=Affinity)
		# Apply algorithm
		fit = ward.fit(X_pca[:, 0:n_features+1])
	else:
		Agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=Linkage, affinity=Affinity)
		# Apply algorithm
		fit = Agg.fit(X_pca[:, 0:n_features+1])

	y_pred = fit.labels_.astype(np.int)


	return X, y_pred, X_pca[:,0:2], params
