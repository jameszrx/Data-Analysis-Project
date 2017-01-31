from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer

stopwords = text.ENGLISH_STOP_WORDS
stemmer = SnowballStemmer("english")

#load the training data
cate = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
datasets = fetch_20newsgroups(subset = 'train', categories = cate, shuffle = True, random_state = 42, remove = ('headers','footers','quotes'))

#load the testing data
cate_tt = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
test_datasets = fetch_20newsgroups(subset = 'test', categories = cate, shuffle = True, random_state = 42, remove = ('headers','footers','quotes'))

#sorting the data into 20 different groups, save them in the "data" list and get the length of each group
index = list()
length = list()
data = list()
for j in range(20):
    index_temp = list()
    index_temp.append(list(np.where(datasets.target == j))[0])
    index.append(index_temp)
    data_temp = list()
    for i in index[j][0]:
        data_temp.append(datasets.data[i])
    data.append(data_temp)
    length.append(len(data_temp))

vectorizer = CountVectorizer(min_df=1)
tfidf_transformer = TfidfTransformer()

#make each class have the same number of items
for j in range(8):
    if j != 3:
        data[j][len(data[3]):] = []

#combine the data of 8 classes into a new list
data_cutoff = list()
for j in range(8):
    data_cutoff.extend(data[j])

#exclude the stop words, punctuations and different stems of a word
data_pro = list()
for j in range(len(data_cutoff)):
    temp = data_cutoff[j]
    temp = re.sub("[^a-zA-Z]"," ",temp)
    temp = temp.lower()
    words = temp.split()
    post_stop = [w for w in words if not w in stopwords]
    post_stem = [stemmer.stem(w1) for w1 in post_stop]
    temp = " ".join(post_stem)
    data_pro.append(temp)

#exclude the stop words, punctuations and different stems of a word for testing set
data_test = test_datasets.data[:]
data_pro_test = list()
for j in range(len(data_test)):
    temp_test = data_test[j]
    temp_test = re.sub("[^a-zA-Z]"," ",temp_test)
    temp_test = temp_test.lower()
    words_test = temp_test.split()
    post_stop_test = [w_test for w_test in words_test if not w_test in stopwords]
    post_stem_test = [stemmer.stem(w1_test) for w1_test in post_stop_test]
    temp_test = " ".join(post_stem_test)
    data_pro_test.append(temp_test)
    
X = vectorizer.fit_transform(data_pro[:])
X_train_tfidf = tfidf_transformer.fit_transform(X)
print(X_train_tfidf.toarray().shape)
