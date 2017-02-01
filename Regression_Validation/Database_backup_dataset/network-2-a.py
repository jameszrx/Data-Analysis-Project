import numpy as np
import csv
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#To run this program in another computer, you should change the path here.
f_w=open('/Users/SoniaYu/Documents/17Winter_LargeScaleDataMining/2a.csv', 'wb') 
with open('/Users/SoniaYu/Documents/17Winter_LargeScaleDataMining/Data/network_backup_dataset.csv', 'rb') as f :
     reader= list(csv.reader(f))
     

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


r = LinearRegression()

#Perform 10-fold cross validation, full length 18588, partition:1859*9+1857*1.
for j in range (0,9) :
    i = j*1858
    mse = 0

    y_cp = list_size[:]
    x_week_cp = list_week[:]
    x_day_cp = list_day[:]
    x_time_cp = list_time[:]
    x_work_flow_cp = list_work_flow[:]
    x_file_cp = list_file[:]
    x_backup_time_cp = list_backup_time[:]

    del y_cp[j*1858:(j+1)*1858]
    del x_week_cp[j*1858:(j+1)*1858]
    del x_day_cp[j*1858:(j+1)*1858]
    del x_time_cp[j*1858:(j+1)*1858]
    del x_work_flow_cp[j*1858:(j+1)*1858]
    del x_file_cp[j*1858:(j+1)*1858]
    del x_backup_time_cp[j*1858:(j+1)*1858]
    
    y1 = y_cp
    x1 = [x_week_cp, x_day_cp, x_time_cp, x_work_flow_cp, x_file_cp, x_backup_time_cp]

    y1_a = np.asarray(y1)
    x1_a = np.asarray(x1).T

    poly = PolynomialFeatures(1)
    x1_p = poly.fit_transform(x1_a)
    y1_p = y1_a

    r.fit(x1_p, y1_p)
    print(r.coef_)
    f_w.write(str(r.coef_))
    f_w.write(',')
    f_w.write('\n')
    
    while i < 1858*(j+1) :
        x_t = np.asarray([list_week[i], list_day[i], list_time[i], list_work_flow[i], list_file[i], list_backup_time[i]]).reshape(1, -1)
        x_t_p = poly.fit_transform(x_t)
        y_prediction = r.predict(x_t_p)
        y_prediction_list = y_prediction.tolist()[0]
        mse = mse + (y_prediction_list - list_size[i])**2
        i = i + 1

    print(math.sqrt(mse/1859))
    f_w.write(str(math.sqrt(mse/1859)))
    f_w.write(',')
    f_w.write('\n')

mse = 0
i = 9*1858
y_cp = list_size[:]
x_week_cp = list_week[:]
x_day_cp = list_day[:]
x_time_cp = list_time[:]
x_work_flow_cp = list_work_flow[:]
x_file_cp = list_file[:]
x_backup_time_cp = list_backup_time[:]
del y_cp[9*1858:]
del x_week_cp[9*1858:]
del x_day_cp[9*1858:]
del x_time_cp[9*1858:]
del x_work_flow_cp[9*1858:]
del x_file_cp[9*1858:]
del x_backup_time_cp[9*1858:]
y1 = y_cp
x1 = [x_week_cp, x_day_cp, x_time_cp, x_work_flow_cp, x_file_cp, x_backup_time_cp]
y1_a = np.asarray(y1)
x1_a = np.asarray(x1).T
poly = PolynomialFeatures(1)
x1_p = poly.fit_transform(x1_a)
y1_p = y1_a
r.fit(x1_p, y1_p)
print(r.coef_)
f_w.write(str(r.coef_))
f_w.write(',')
f_w.write('\n')
fitted_list = list()

while i < 18588 :
    x_t = np.asarray([list_week[i], list_day[i], list_time[i], list_work_flow[i], list_file[i], list_backup_time[i]]).reshape(1, -1)
    x_t_p = poly.fit_transform(x_t)
    y_prediction = r.predict(x_t_p)
    y_prediction_list = y_prediction.tolist()[0]
    fitted_list.append(y_prediction_list)
    mse = mse + (y_prediction_list - list_size[i])**2
    i = i + 1

    
print(math.sqrt(mse/1867))
f_w.write(str(math.sqrt(mse/1867)))
f_w.write(',')
f_w.write('\n')

f_w.close()

#Plot fitted values and actual values scattered plot over time
plt.figure()
plt.scatter(range(1866), list_size[9*1858:], marker = 'x', color = 'r',  label='Actual Values')
plt.hold(True)
plt.scatter(range(1866), fitted_list, marker = '*', color = 'b',  label='Fitted Values')
plt.axis([0, 2000, -0.2, 1.2])
plt.title('Fitted values and actual values scattered plot over time')
plt.show()

#Plot fitted values versus actual values
plt.figure()
plt.scatter(list_size[9*1858:], fitted_list,marker='+',color='r')
plt.hold(True)
plt.plot([-0.2,1.3],[-0.2,1.3],'b-',linewidth = 2.0)
plt.axis([-0.2, 1.3, -0.2, 1.3])
plt.ylabel('Fitted values')
plt.xlabel('Actual values')
plt.title('Fitted values versus actual values')
plt.show()

#Plot residual versus fitted values plot'
plt.figure()
plt.scatter(fitted_list, (np.array(list_size[9*1858:]) - np.array(fitted_list)).tolist(),marker='+',color='r')
plt.hold(True)
plt.plot([-0.05,0.35],[0,0],'b-',linewidth = 2.0)
plt.axis([-0.05, 0.35, -0.2, 0.8])
plt.ylabel('Residual values')
plt.xlabel('Fitted values')
plt.title('Residual versus fitted values plot')
plt.show()


