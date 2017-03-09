# coding: utf-8

from sklearn.datasets import fetch_20newsgroups

categories_C =['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
categories_R =['rec.autos','rec.motorcycles', 'rec.sport.baseball' ,'rec.sport.hockey']
C_train = fetch_20newsgroups(subset='train', categories=categories_C, shuffle=True, random_state= 42)
C_test = fetch_20newsgroups(subset='test', categories=categories_C, shuffle=True, random_state= 42)
R_train = fetch_20newsgroups(subset='train', categories=categories_R, shuffle=True, random_state= 42)
R_test = fetch_20newsgroups(subset='test', categories=categories_R, shuffle=True, random_state= 42)
T_train= fetch_20newsgroups(subset='train', categories=categories_C+categories_R, shuffle=True, random_state= 42)

print ("Size of data: ")
print (len(T_train.data))
print ("Size of target: ")
print (len(T_train.target))
print("")


#If don't remove stop words and dif stems of word
from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = CountVectorizer(min_df=1)
vectorizer =TfidfVectorizer()
test_data=T_train.data
#print test_data
X = vectorizer.fit_transform(test_data)

ori_samples, ori_features = X.shape
print("Number of features before excluding stopwords and stemmers")
print("#samples: %d, #features: %d" % (ori_samples, ori_features))
print("")




from sklearn.feature_extraction import text
stop_words = text.ENGLISH_STOP_WORDS
#print (stop_words)


#Exclude stem and stopwords from the training dataset  
test_data=T_train.data
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(
    min_df=1, stop_words='english', decode_error='ignore')

#X_train is the result after tfidf processing
X_train = vectorizer.fit_transform(test_data)

num_samples, num_features = X_train.shape
print("Number of features after excluding stopwords and stemmers")
print("#samples: %d, #features: %d" % (num_samples, num_features))

target_train = []
for doc in T_train.target:
    if T_train.target_names[doc] in categories_C :
        target_train.append(0);
    else:
        target_train.append(1);

#Calculate the confusion matrix of KMeans
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Computer technology', 'Recreational activity'], rotation=45)
    plt.yticks(tick_marks, ['Computer technology', 'Recreational activity'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



y_pred = KMeans(n_clusters=2).fit_predict(X_train)
y_true = target_train


cm = confusion_matrix(y_true, y_pred);
np.set_printoptions(precision=2)
print("")
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)
plt.show()



#Calculation of different scores
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

homo_score = homogeneity_score(target_train, y_pred)
print("")
print("The homogeneity score of KMeans cluster is:")
print(homo_score)

comp_score = completeness_score(target_train, y_pred)
print("")
print("The completeness score of KMeans cluster is:")
print(comp_score)

rand_score = adjusted_rand_score(target_train, y_pred)
print("")
print("The adjusted rand score of KMeans cluster is:")
print(rand_score)

info_score = adjusted_mutual_info_score(target_train, y_pred)
print("")
print("The adjusted mutual info score of KMeans cluster is:")
print(info_score)