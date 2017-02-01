# refer:
# http://www.programcreek.com/python/example/83247/sklearn.cross_validation.KFold
# https://onlinecourses.science.psu.edu/stat501/node/310
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, cross_validation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

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

feature_wf=[]
result_wf=[]

for line in lines:
    tmp=[]
    p=line.split(',');
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
fixed=[]

reg=linear_model.LinearRegression()

for degree in range(1,7):
    reg=make_pipeline(PolynomialFeatures(degree), LinearRegression())

    score=-cross_validation.cross_val_score(reg, feature_wf, result_wf,  cv=10, scoring='mean_squared_error')
    
    fixed.append(np.mean(np.sqrt(score[0])))
    rmse.append(np.mean(np.sqrt(score)))

print 'fixed training and test set:', fixed
print 'average rmse:', rmse

plt.figure()
plt.plot(range(1,7), fixed, 'r')
plt.plot(range(1,7), rmse, 'g')
plt.grid(True)
plt.title('Polynomial Fitting')
plt.xlabel('degree')
plt.ylabel('fixed set & avg rmse')
plt.show()



