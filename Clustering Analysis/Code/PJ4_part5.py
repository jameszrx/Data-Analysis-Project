# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from time import time
import numpy as np


#categories = [
#    'alt.atheism',
#    'talk.religion.misc',
#    'comp.graphics',
#    'sci.space',
#]
categories = None

print("Loading 20 newsgroups dataset for 20 categories.")

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X = vectorizer.fit_transform(dataset.data)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

#Change dim list to find proper dimension
dim = [2, 5, 10, 20, 30, 40, 50, 100]

#LAS dimensionality reduction
print("Performing dimensionality reduction using LSA (without normaliazation)")
for m in dim:
    t0 = time()
    svd = TruncatedSVD(n_components=m, random_state=42)
    X_dim = svd.fit_transform(X)
    print("For dimension %d" %m)
    print("Dimension reduction done in %fs" % (time() - t0))
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    t0 = time()
    km.fit(X_dim)
    print("Kmeans clustering done in %0.3fs" % (time() - t0))
    print("Homogeneity score: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness score: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("Adjusted Rand-Index score: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Adjusted mutual info score: %0.3f"
          % metrics.adjusted_mutual_info_score(labels, km.labels_))
    print()


#LSA dimensionality reduction with normalization
print("Performing dimensionality reduction using LSA")
# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
for m in dim:
    t0 = time()
    svd = TruncatedSVD(n_components=m, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X_dim = lsa.fit_transform(X)
    print("For dimension %d" %m)
    print("Dimension reduction done in %fs" % (time() - t0))
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    t0 = time()
    km.fit(X_dim)
    print("Kmeans clustering done in %0.3fs" % (time() - t0))
    print("Homogeneity score: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness score: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("Adjusted Rand-Index score: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Adjusted mutual info score: %0.3f"
          % metrics.adjusted_mutual_info_score(labels, km.labels_))
    print()


#LSA with log
for m in dim:
    t0 = time()
    svd = TruncatedSVD(n_components=m, random_state=42)
    X_dim = svd.fit_transform(X)
    #log
    min = X_dim.min()
    diff = 0.01 - min
    num_sample_svd, num_feature_svd = X_dim.shape
    for i in range(num_sample_svd):
          for j in range(num_feature_svd):
              X_dim[i][j] = diff + X_dim[i][j]
    X_log = np.log(X_dim)
    print("For dimension %d" %m)
    print("Dimension reduction done in %fs" % (time() - t0))
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    t0 = time()
    km.fit(X_log)
    print("Kmeans clustering done in %0.3fs" % (time() - t0))
    print("Homogeneity score: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness score: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("Adjusted Rand-Index score: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Adjusted mutual info score: %0.3f"
          % metrics.adjusted_mutual_info_score(labels, km.labels_))
    print()


#NMF dimensionality reduction
print("Performing dimensionality reduction using NMF")
for m in dim:
    t0 = time()
    nmf = NMF(n_components=m, init='random', random_state=42)
    X_dim = nmf.fit_transform(X)
    print("For dimension %d" %m)
    print("Dimension reduction done in %fs" % (time() - t0))
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    t0 = time()
    km.fit(X_dim)
    print("Kmeans clustering done in %0.3fs" % (time() - t0))
    print("Homogeneity score: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness score: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("Adjusted Rand-Index score: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Adjusted mutual info score: %0.3f"
          % metrics.adjusted_mutual_info_score(labels, km.labels_))
    print()


#NMF dimensionality reduction with normalization
print("Performing dimensionality reduction using NMF")
for m in dim:
    t0 = time()
    nmf = NMF(n_components=m, init='random', random_state=42)
    normalizer = Normalizer(copy=False)
    norm_nmf= make_pipeline(nmf, normalizer)
    X_dim = norm_nmf.fit_transform(X)
    print("For dimension %d" %m)
    print("Dimension reduction done in %fs" % (time() - t0))
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    t0 = time()
    km.fit(X_dim)
    print("Kmeans clustering done in %0.3fs" % (time() - t0))
    print("Homogeneity score: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness score: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("Adjusted Rand-Index score: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Adjusted mutual info score: %0.3f"
          % metrics.adjusted_mutual_info_score(labels, km.labels_))
    print()


#NMF dimensionality reduction with log
print("Performing dimensionality reduction using NMF")
for m in dim:
    t0 = time()
    nmf = NMF(n_components=m, init='random', random_state=42)
    X_dim = nmf.fit_transform(X)
    min = X_dim.min()
    diff = 0.01 - min
    num_sample_svd, num_feature_svd = X_dim.shape
    for i in range(num_sample_svd):
          for j in range(num_feature_svd):
              X_dim[i][j] = diff + X_dim[i][j]
    X_log = np.log(X_dim)
    print("For dimension %d" %m)
    print("Dimension reduction done in %fs" % (time() - t0))
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    t0 = time()
    km.fit(X_log)
    print("Kmeans clustering done in %0.3fs" % (time() - t0))
    print("Homogeneity score: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness score: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("Adjusted Rand-Index score: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Adjusted mutual info score: %0.3f"
          % metrics.adjusted_mutual_info_score(labels, km.labels_))
    print()


#Code to find approriate k in k-means clustering
#change k_value list to find proper k
k_value = [2, 3, 4, 5]

#Truncted SVD with normalization at dimension=10
print("Truncted SVD with normalization at dimension=10:")
for k in k_value:
    t0 = time()
    svd = TruncatedSVD(n_components=10, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X_dim = lsa.fit_transform(X)
    print("For k = %d" %k)
    print("Dimension reduction done in %fs" % (time() - t0))
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    t0 = time()
    km.fit(X_dim)
    print("Kmeans clustering done in %0.3fs" % (time() - t0))
    print("Homogeneity score: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness score: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("Adjusted Rand-Index score: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Adjusted mutual info score: %0.3f"
          % metrics.adjusted_mutual_info_score(labels, km.labels_))
    print()

    

#NMF with logarithm at dimension=30
print("NMF with logarithm at dimension=30:")
for k in k_value:
    t0 = time()
    nmf = NMF(n_components=30, init='random', random_state=42)
    X_dim = nmf.fit_transform(X)
    min = X_dim.min()
    diff = 0.01 - min
    num_sample_svd, num_feature_svd = X_dim.shape
    for i in range(num_sample_svd):
          for j in range(num_feature_svd):
              X_dim[i][j] = diff + X_dim[i][j]
    X_log = np.log(X_dim)
    print("For k = %d" %k)
    print("Dimension reduction done in %fs" % (time() - t0))
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    t0 = time()
    km.fit(X_log)
    print("Kmeans clustering done in %0.3fs" % (time() - t0))
    print("Homogeneity score: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness score: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("Adjusted Rand-Index score: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Adjusted mutual info score: %0.3f"
          % metrics.adjusted_mutual_info_score(labels, km.labels_))
    print()

