import csv
import math
import random
import numpy as np
import pandas as pd
from math import sqrt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pybrain.structure import TanhLayer
from sklearn import datasets, linear_model
from pybrain.structure import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets.supervised import SupervisedDataSet as SDS

#To run this program in another computer, you should change the path here.
with open('/Users/SoniaYu/Documents/17Winter_LargeScaleDataMining/Data/network_backup_dataset.csv', 'rb') as f :
     reader = list(csv.reader(f))

data = reader[1 : ]
random.shuffle(data)

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
    
X = (list_week,list_day,list_time,list_work_flow, list_file, list_backup_time,list_backup_time,list_size)
X=np.mat(X).T


x_test = X[:1859,0:-2]
y_test = X[:1859,-1]

x_train = X[1859:,0:-2]
y_train = X[1859:,-1]

list_prediction = list()
#input_size = x_train.shape[1]  #column number
#target_size = y_train.shape[1]
input_size = 6
target_size = 1
hidden0_size = 10
hidden1_size = 3
epochs = 500
mse = 0

#Prepare the training and testing dataset
ds = SDS(input_size,target_size)
ds.setField('input',x_train)
ds.setField('target',y_train)

#Init the network and train
net = buildNetwork(input_size,hidden0_size, target_size, bias = True,hiddenclass=TanhLayer)
trainer = BackpropTrainer(net, ds)
#trainer.trainUntilConvergence(maxEpochs = 1000)

for i in range( epochs ):
	mse_t = trainer.train()
	rmse_t = sqrt( mse_t )
	print ('training RMSE, epoch {}: {}'.format( i + 1, rmse_t ))

for i in range(1859):
        y_prediction = net.activate(np.array(x_test[i,:])[0])
        list_prediction.append(y_prediction)
        mse = mse + (y_prediction - y_test[i,0])*(y_prediction - y_test[i,0])
    
mse=mse/1859
rmse=np.sqrt(mse)
print(rmse)
x = range(0, 1859)
plt.scatter(x,list_prediction,color='r', marker = 'x',label="Prediction Value")
plt.hold(True)
plt.grid(True)
plt.scatter(x,list_size[:1859],color = 'b',marker = '+',label="Actual Value")
plt.title('Prediction by Neural Network Regression Model')
plt.ylabel('Backup Size')
plt.xlabel('File Number')
plt.axis([0,2000,-0.2,1.2])
plt.legend()
plt.show()
