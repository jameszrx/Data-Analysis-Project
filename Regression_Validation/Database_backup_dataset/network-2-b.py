import csv
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor

#To run this program in another computer, you should change the path here.
with open('/Users/SoniaYu/Documents/17Winter_LargeScaleDataMining/Data/network_backup_dataset.csv', 'rb') as f :
     reader = list(csv.reader(f))

data = reader[1 : ]

list_week = list()
list_day = list()
list_time = list()
list_work_flow = list()
list_file = list()
list_size = list()
list_backup_time = list()



i = 0
while i <= len(data) - 1 :
    list_size.append(float(data[i][5]))
    list_backup_time.append(float(data[i][6]))

    j=data[i][0]
    list_week.append(int(j))

    k=data[i][1]
    if k == 'Monday' :
        list_day.append(1)
    elif k == 'Tuesday' :
        list_day.append(2)
    elif k == 'Wednesday' :
        list_day.append(3)
    elif k == 'Thursday' :
        list_day.append(4)
    elif k == 'Friday' :
        list_day.append(5)
    elif k == 'Saturday' :
        list_day.append(6)
    elif k == 'Sunday' :
        list_day.append(7)

    l=data[i][2]
    list_time.append(int(l))
          
    m=data[i][3]
    list_work_flow.append(int(m[10])+1)

    n=data[i][4]
    if len(n)==6:
        list_file.append(int(n[5])+1)
    elif len(n)==7:
        list_file.append(int(n[5:7])+1)

    i = i + 1

y = list_size[:]
x1=list_week[:]
x2=list_day[:]
x3=list_time[:]
x4=list_work_flow[:]
x5=list_file[:]
x6=list_backup_time[:]

X = [x1, x2, x3, x4, x5, x6]
X=np.asarray(X).T
list_rmse=list()

'''
for depth in range(4,11):
for estimator in [1,10,15,20]:
    r = RandomForestRegressor(n_estimators=estimator, max_features=6, max_depth=10)
    scores=cross_validation.cross_val_score(r,X,y,cv=10, scoring='mean_squared_error')
    y_predict=cross_validation.cross_val_predict(r,X,y,cv=10)
    rmse=math.sqrt(-np.mean(scores))
    list_rmse.append(rmse)
    print(rmse)

plt.scatter([1,10,15,20],list_rmse,color='r')
plt.hold(True)
plt.xlabel('Number of trees')
plt.ylabel('RMSE')
plt.title('Fitted Values vs Actual Values')
plt.axis([0, 20, 0, 0.10])
plt.show()

##plt.scatter(range(4,11),list_rmse,color='r')
##plt.hold(True)
##plt.show()
'''

#Use Rodom Forest Model
r = RandomForestRegressor(n_estimators=20, max_features=6, max_depth=10)
scores=cross_validation.cross_val_score(r,X,y,cv=10, scoring='mean_squared_error')
y_predict=cross_validation.cross_val_predict(r,X,y,cv=10)
rmse=math.sqrt(-np.mean(scores))
list_rmse.append(rmse)

#Plot 
plt.scatter(list_size,y_predict,marker='+',color='r')
plt.hold(True)
plt.plot([-0.2,1.2],[-0.2,1.2],linewidth=2.0)
plt.axis([-0.2,1.2, -0.2, 1.2])
plt.xlabel('Actual values')
plt.ylabel('Fitted values')
plt.title('Fitted values versus actual values')
plt.show()


