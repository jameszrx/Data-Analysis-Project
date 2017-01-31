from sklearn.datasets import fetch_20newsgroups
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from sklearn import metrics

data_tr_r = np.loadtxt('multitest_out.csv', delimiter = ',')
data_ts_r = np.loadtxt('multitrain_out.csv', delimiter = ',')

data_tr = data_tr_r[:, :-1]
data_ts = data_ts_r[:, :-1]
label_tr = data_tr_r[:,-1]
label_ts = data_ts_r[:,-1]

n_classes = label_tr.shape[0]

# Learn to predict each class against one class
clf = OneVsRestClassifier(LinearSVC(random_state = 0))
OvsO = clf.fit(data_tr, label_tr)

result=clf.predict(data_ts)
#accuracy=clf.score(data_ts,label_ts)
accuracy = metrics.accuracy_score(result, label_ts)
error_vector = result - label_ts
error = 0

p_data= clf.fit(data_tr, label_tr).decision_function(data_ts)
p_data=p_data[:,1]

conf_mat=metrics.confusion_matrix(label_ts,result)
precision=metrics.precision_score(label_ts, result, average = None)
recall=metrics.recall_score(label_ts, result, average = None)
print ("confusion_matrix:")
print (conf_mat)
print ("precision:", precision)
print ("recall:", recall)
print ("accuracy:", accuracy)
