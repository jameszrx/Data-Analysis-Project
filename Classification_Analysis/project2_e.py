import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data_tr_r = np.genfromtxt('train_out.csv', delimiter = ',')
data_ts_r = np.genfromtxt('test_out.csv', delimiter = ',')

data_tr = data_tr_r[:, :-1]
data_ts = data_ts_r[:, :-1]
label_tr = data_tr_r[:,-1]
label_ts = data_ts_r[:,-1]

clf = SVC(C = 1.0, kernel = 'linear')
svm = clf.fit(data_tr, label_tr)

w = svm.coef_
b = svm.intercept_
weight = np.dot(data_ts,w.T) + b
#result = clf.predict(data_ts)
TPR = list()
FPR = list()
confusion_matrix = list()
accuracy = list()
precision = list()
recall = list()
for thre in np.arange(-3, 3.1, 0.1):
    result = (np.sign(weight - thre) + 1) / 2
    result = result[:,0]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for j in range(len(result)):
        if label_ts[j] == 1 and result[j] == 1:
            TP += 1
        elif label_ts[j] == 0 and result[j] == 0:
            TN += 1
        elif label_ts[j] == 0 and result[j] == 1:
            FP += 1
        else:
            FN += 1
    TPR.append(TP / (TP + FN))
    FPR.append(FP / (FP + TN))
    confusion_matrix.append(np.array([[TP, FN], [FP, TN]]))
    error_vector = result - label_ts
    error = 0
    for j in range(len(result)):
        if error_vector[j] == -1:
            error = error + 1
        else:
            error = error + error_vector[j]
    accuracy.append(1 - error/len(result))
    precision.append(TP / (TP + FP))
    recall.append(TP / (TP + FN))
    
plt.figure()
plt.plot(FPR, TPR, 'r-', linewidth = 4.0)
plt.axis([-0.05, 0.95, 0, 1.05])
plt.title('ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

print('the accuracy is:')
print(accuracy[31])
print('the precision is:')
print(precision[31])
print('the recall is:')
print(recall[31])
print('the confusion matrix is:')
print(confusion_matrix[31])
