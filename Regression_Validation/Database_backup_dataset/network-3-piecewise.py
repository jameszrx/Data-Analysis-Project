# refer:
# http://www.programcreek.com/python/example/83247/sklearn.cross_validation.KFold
# https://onlinecourses.science.psu.edu/stat501/node/310
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, cross_validation


def weekday(str):
    if(str=='Monday'):
        return 1;
    elif(str=='Tuesday'):
    	return 2;
    elif(str=='Wednesday'):
        return 3;
    elif(str=='Thursday'):
    	return 4;
    elif(str=='Friday'):
    	return 5;
    elif(str=='Saturday'):
        return 6;
    else:
    	return 7;


def getnumber(str):
    p=str.split('_');
    return (float)(p[len(p)-1]);

def select(dataset, index):
    res=[]
    for x in index:
        res.append(dataset[x])
    return res


f=open('/Users/kay/Downloads/network_backup_dataset.csv','r')
lines=f.readlines()[2:18590]
f.close()

for workflow in range(5):
    feature_wf=[]
    result_wf=[]

    for line in lines:
        tmp=[]
        p=line.split(',');
        if(getnumber(p[3])==workflow):
            tmp.append(float(p[0]))
            tmp.append(weekday(p[1]))
            tmp.append(float(p[2]))
            tmp.append(getnumber(p[3]))
            tmp.append(getnumber(p[4]))
            tmp.append(float(p[6]))
            feature_wf.append(tmp)
            result_wf.append(float(p[5]))

    kf = cross_validation.KFold(len(feature_wf), 10, True)

    rmse=[]
    score=[]

    reg=linear_model.LinearRegression()

    for train, test in kf:
        feature_wf_train=select(feature_wf, train)
        feature_wf_test=select(feature_wf, test)
        result_wf_train=select(result_wf, train)
        result_wf_test=select(result_wf, test)

        reg.fit(feature_wf_train, result_wf_train)
    	
    predict=reg.predict(feature_wf)

    rmse.append(np.sqrt((predict - result_wf) ** 2).mean())
    score.append(reg.score(feature_wf, result_wf))

    print 'workflow id:', workflow
    print rmse

    plt.show()
