import nltk.stem
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

categories_C =['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
categories_R =['rec.autos','rec.motorcycles', 'rec.sport.baseball' ,'rec.sport.hockey']
C_train = fetch_20newsgroups(subset='train', categories=categories_C, shuffle=True, random_state= 42)
C_test = fetch_20newsgroups(subset='test', categories=categories_C, shuffle=True, random_state= 42)
R_train = fetch_20newsgroups(subset='train', categories=categories_R, shuffle=True, random_state= 42)
R_test = fetch_20newsgroups(subset='test', categories=categories_R, shuffle=True, random_state= 42)
T_train= fetch_20newsgroups(subset='train', categories=categories_C+categories_R, shuffle=True, random_state= 42)

#If don't remove stop words and dif stems of word
from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = CountVectorizer(min_df=1)
vectorizer =TfidfVectorizer()
test_data=T_train.data
#print test_data
X = vectorizer.fit_transform(test_data)
X.shape
print (X.shape)

stop_words = text.ENGLISH_STOP_WORDS
#print (stop_words)

#Exclude stem and stopwords from the training dataset  
test_data=T_train.data
english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(
    min_df=1, stop_words='english', decode_error='ignore')

#X_train is the result after tfidf processing
X_train = vectorizer.fit_transform(test_data)

target_train = []
for doc in T_train.target:
    if T_train.target_names[doc] in categories_C :
        target_train.append(0);
    else:
        target_train.append(1);

n_component = 2

nmf = NMF(n_component)
X_nmf = nmf.fit_transform(X_train)
num_samples_nmf, num_features_nmf = X_nmf.shape
print("")
print("NMF:  # of features: %d" % (num_features_nmf))

y_pred = KMeans(n_clusters=2).fit_predict(X_nmf)


#Transit the matrix
min = X_nmf.min()
difference = 0.008 - min
for i in range(num_samples_nmf):
    for j in range(num_features_nmf):
        X_nmf[i][j] = difference + X_nmf[i][j]
#Take logarithm
X_log = np.log(X_nmf)
y_pred = KMeans(n_clusters=2).fit_predict(X_log)
 
homo_score = homogeneity_score(target_train, y_pred)
print("The homogeneity score of KMeans cluster after NMF & logarithm is:")
print(homo_score)

comp_score = completeness_score(target_train, y_pred)
print("The completeness score of KMeans cluster after NMF & logarithm is:")
print(comp_score)

rand_score = adjusted_rand_score(target_train, y_pred)
print("The adjusted rand score of KMeans cluster after NMF & logarithm is:")
print(rand_score)

info_score = adjusted_mutual_info_score(target_train, y_pred)
print("The adjusted mutual info score of KMeans cluster after NMF & logarithm is:")
print(info_score)

x = []
y = []
for i in range(num_samples_nmf):
    x.append(X_log[i][0])
    y.append(X_log[i][1])

plt.figure()
plt.subplot(211)     
plt.scatter(x, y, marker = 'o', c = y_pred)
plt.title("Predicting Labels")
plt.subplot(212)
plt.scatter(x, y, marker = 'o', c = target_train)
plt.title("Actual Labels")
plt.show()
