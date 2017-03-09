# coding: utf-8

from sklearn.datasets import fetch_20newsgroups

categories_C =['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
categories_R =['rec.autos','rec.motorcycles', 'rec.sport.baseball' ,'rec.sport.hockey']
C_train = fetch_20newsgroups(subset='train', categories=categories_C, shuffle=True, random_state= 42)
C_test = fetch_20newsgroups(subset='test', categories=categories_C, shuffle=True, random_state= 42)
R_train = fetch_20newsgroups(subset='train', categories=categories_R, shuffle=True, random_state= 42)
R_test = fetch_20newsgroups(subset='test', categories=categories_R, shuffle=True, random_state= 42)
T_train= fetch_20newsgroups(subset='train', categories=categories_C+categories_R, shuffle=True, random_state= 42)

print (len(C_train.data))
print (len(T_train.data))


#If don't remove stop words and dif stems of word
from sklearn.feature_extraction.text import TfidfVectorizer
#vectorizer = CountVectorizer(min_df=1)
vectorizer =TfidfVectorizer()
test_data=T_train.data
#print test_data
X = vectorizer.fit_transform(test_data)
X.shape
print (X.shape)



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
print("#samples: %d, #features: %d" % (num_samples, num_features))