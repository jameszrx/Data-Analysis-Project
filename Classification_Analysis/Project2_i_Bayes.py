import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt

data_tr_r = np.loadtxt('multitest_out.csv', delimiter = ',')
data_ts_r = np.loadtxt('multitrain_out.csv', delimiter = ',')

data_tr = data_tr_r[:, :-1]
data_ts = data_ts_r[:, :-1]
label_tr = data_tr_r[:,-1]
label_ts = data_ts_r[:,-1]

clf = GaussianNB()
GNB = clf.fit(data_tr, label_tr)

result=clf.predict(data_ts)
#accuracy=clf.score(data_ts,label_ts)
accuracy = metrics.accuracy_score(result, label_ts)
error_vector = result - label_ts
error = 0

conf_mat=metrics.confusion_matrix(label_ts,result)
precision=metrics.precision_score(label_ts, result, average = None)
recall=metrics.recall_score(label_ts, result, average = None)
print ("confusion_matrix:")
print (conf_mat)
print ("precision:", precision)
print ("recall:", recall)
print ("accuracy:", accuracy)