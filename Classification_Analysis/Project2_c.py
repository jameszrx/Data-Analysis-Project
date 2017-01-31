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
cate = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
datasets = fetch_20newsgroups(subset = 'train', categories = cate, shuffle = True, random_state = 42, remove = ('headers','footers','quotes'))

#load the testing data
cate_tt = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
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

#data processing for TFxICF
data_icf = list()
for j in range(20):
    data_cl = ""
    for i in range(len(data[j])):
        data_cl = data_cl + " " + data[j][i]
    data_icf.append(data_cl)
data_pro_icf = list()
for j in range(len(data_icf)):
    temp_icf = data_icf[j]
    temp_icf = re.sub("[^a-zA-Z]"," ",temp_icf)
    temp_icf = temp_icf.lower()
    words_icf = temp_icf.split()
    post_stop_icf = [w_icf for w_icf in words_icf if not w_icf in stopwords]
    post_stem_icf = [stemmer.stem(w1_icf) for w1_icf in post_stop_icf]
    temp_icf = " ".join(post_stem_icf)
    data_pro_icf.append(temp_icf)
X_icf = vectorizer.fit_transform(data_pro_icf[:])
X_train_icf = tfidf_transformer.fit_transform(X_icf)
ibm_li = X_train_icf.toarray()[3]
mac_li = X_train_icf.toarray()[4]
forsale_li = X_train_icf.toarray()[6]
christian_li = X_train_icf.toarray()[15]

ibm_s = sorted(ibm_li)
mac_s = sorted(mac_li)
forsale_s = sorted(forsale_li)
christian_s = sorted(christian_li)

ibm_s = ibm_s[-10:]
mac_s = mac_s[-10:]
forsale_s = forsale_s[-10:]
christian_s = christian_s[-10:]

ibm_index = list()
mac_index = list()
forsale_index = list()
christian_index = list()

for j in range(len(ibm_li)):
    if ibm_li[j] in ibm_s:
        ibm_index.append(j)
    if mac_li[j] in mac_s:
        mac_index.append(j)
    if forsale_li[j] in forsale_s:
        forsale_index.append(j)
    if christian_li[j] in christian_s:
        christian_index.append(j)

ibm_features = list()
mac_features = list()
forsale_features = list()
christian_features = list()
for j in ibm_index:
    ibm_features.append(vectorizer.get_feature_names()[j])
for j in mac_index:
    mac_features.append(vectorizer.get_feature_names()[j])
for j in forsale_index:
    forsale_features.append(vectorizer.get_feature_names()[j])
for j in christian_index:
    christian_features.append(vectorizer.get_feature_names()[j])
print("10 most important terms for ibm:")
print(ibm_features)
print("10 most important terms for mac:")
print(mac_features)
print("10 most important terms for forsale:")
print(forsale_features)
print("10 most important terms for christian:")
print(christian_features)
