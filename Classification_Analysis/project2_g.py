import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt

data_tr_r = np.loadtxt('train_out.csv', delimiter = ',')
data_ts_r = np.loadtxt('test_out.csv', delimiter = ',')

data_tr = data_tr_r[:, :-1]
data_ts = data_ts_r[:, :-1]
label_tr = data_tr_r[:,-1]
label_ts = data_ts_r[:,-1]

clf = GaussianNB()
GNB = clf.fit(data_tr, label_tr)



result=clf.predict(data_ts)
accuracy=clf.score(data_ts,label_ts)

p_data=clf.predict_proba(data_ts)
print(result)
p_data=p_data[:,1]
fpr,tpr,thresholds=metrics.roc_curve(label_ts,p_data)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

conf_mat=metrics.confusion_matrix(label_ts,result)
precision=metrics.precision_score(label_ts,result)
recall=metrics.recall_score(label_ts,result)
print(conf_mat)
print(precision)
print(recall)
print(accuracy)

