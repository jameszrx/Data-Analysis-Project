import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data_tr_r = np.genfromtxt('D:\\study\\Winter 2016\\Big Data\\Project 2\\data\\train_out.csv', delimiter = ',')
data_ts_r = np.genfromtxt('D:\\study\\Winter 2016\\Big Data\\Project 2\\data\\test_out.csv', delimiter = ',')

data_tr = data_tr_r[:, :-1]
data_ts = data_ts_r[:, :-1]
label_tr = data_tr_r[:,-1]
label_ts = data_ts_r[:,-1]

data_tr_f = list()
label_tr_f = list()
data_ts_f = list()
label_ts_f = list()

data_tr_f.append(data_tr[925:])
label_tr_f.append(label_tr[925:])
data_tr_f.append(np.concatenate((data_tr[0:925], data_tr[1850:])))
label_tr_f.append(np.concatenate((label_tr[0:925], label_tr[1850:])))
data_tr_f.append(np.concatenate((data_tr[0:1850], data_tr[2775:])))
label_tr_f.append(np.concatenate((label_tr[0:1850], label_tr[2775:])))
data_tr_f.append(np.concatenate((data_tr[0:2775], data_tr[3700:])))
label_tr_f.append(np.concatenate((label_tr[0:2775], label_tr[3700:])))
data_tr_f.append(data_tr[0:3700])
label_tr_f.append(label_tr[0:3700])

for j in range(4):
    data_ts_f.append(data_tr[j*925:j*925 + 924])
    label_ts_f.append(label_tr[j*925:j*925 + 924])
data_ts_f.append(data_tr[3700:])
label_ts_f.append(label_tr[3700:])


confusion_matrix = list()
accuracy = list()
precision = list()
recall = list()
accuracy = list()
for gamma in [1000, 100, 10, 1, 0.1, 0.01, 0.001]:
    clf = SVC(C = gamma, kernel = 'linear')
    TP_li = list()
    TN_li = list()
    FP_li = list()
    FN_li = list()
    accu = 0
    for j in range(5):
        svm = clf.fit(data_tr_f[j], label_tr_f[j])
        result = clf.predict(data_ts_f[j])
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(result)):
            if label_ts_f[j][i] == 1 and result[i] == 1:
                TP += 1
            elif label_ts_f[j][i] == 0 and result[i] == 0:
                TN += 1
            elif label_ts_f[j][i] == 0 and result[i] == 1:
                FP += 1
            else:
                FN += 1

        error_vector = result - label_ts_f[j]
        error = 0
        for i in range(len(result)):
            if error_vector[i] == -1:
                error = error + 1
            else:
                error = error + error_vector[i]

        accu += 1 - error/len(result)

    accuracy.append(accu/5)
            
    TP_li.append(TP)
    TN_li.append(TN)
    FP_li.append(FP)
    FN_li.append(FN)

    TP_s = 0
    TN_s = 0
    FP_s = 0
    FN_s = 0
    for j in range(len(TP_li)):
        TP_s += TP_li[j]
        TN_s += TN_li[j]
        FP_s += FP_li[j]
        FN_s += FN_li[j]

    TP_s = TP_s / 5
    TN_s = TN_s / 5
    FP_s = FP_s / 5
    FN_s = FN_s / 5
    confusion_matrix.append(np.array([[TP_s, FN_s], [FP_s, TN_s]]))
    if gamma != 0.01 and gamma != 0.001:
        precision.append(TP_s / (TP_s + FP_s))
    recall.append(TP_s / (TP_s + FN_s))

clf = SVC(C = 1000, kernel = 'linear')
svm = clf.fit(data_tr, label_tr)
result = clf.predict(data_ts)
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

error_vector = result - label_ts
error = 0
for j in range(len(result)):
    if error_vector[j] == -1:
        error = error + 1
    else:
        error = error + error_vector[j]

accuracy.append(1 - error/len(result))
confusion_matrix.append(np.array([[TP, FN], [FP, TN]]))
precision.append(TP / (TP + FP))
recall.append(TP / (TP + FN))

print('the accuracy is:')
print(accuracy[-1])
print('the recall is:')
print(recall[-1])
print('the precision is:')
print(precision[-1])
print('the confusion matrix is:')
print(confusion_matrix[-1])
