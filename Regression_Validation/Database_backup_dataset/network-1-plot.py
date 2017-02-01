import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path='/Users/kay/Downloads/network_backup_dataset.csv'
rawdata=pd.read_csv(path)

f=open('/Users/kay/Downloads/network_backup_dataset.csv','r')
lines=f.readlines()[2:18590]
f.close()

workflow_0_file_0=[]
workflow_0_file_1=[]
workflow_0_file_2=[]
workflow_0_file_3=[]
workflow_0_file_4=[]
workflow_0_file_5=[]

workflow_1_file_0=[]
workflow_1_file_1=[]
workflow_1_file_2=[]
workflow_1_file_3=[]
workflow_1_file_4=[]
workflow_1_file_5=[]

workflow_2_file_0=[]
workflow_2_file_1=[]
workflow_2_file_2=[]
workflow_2_file_3=[]
workflow_2_file_4=[]
workflow_2_file_5=[]

workflow_3_file_0=[]
workflow_3_file_1=[]
workflow_3_file_2=[]
workflow_3_file_3=[]
workflow_3_file_4=[]
workflow_3_file_5=[]

workflow_4_file_0=[]
workflow_4_file_1=[]
workflow_4_file_2=[]
workflow_4_file_3=[]
workflow_4_file_4=[]
workflow_4_file_5=[]

for line in lines:
    p=line.split(',')
    if p[3]=='work_flow_0':
        if p[4]=='File_0':
            workflow_0_file_0.append(float(p[5]))
        elif p[4]=='File_1':
            workflow_0_file_1.append(float(p[5]))
        elif p[4]=='File_2':
            workflow_0_file_2.append(float(p[5]))
        elif p[4]=='File_3':
            workflow_0_file_3.append(float(p[5]))
        elif p[4]=='File_4':
            workflow_0_file_4.append(float(p[5]))
        else:
            workflow_0_file_5.append(float(p[5]))
    elif p[3]=='work_flow_1':
        if p[4]=='File_6':
            workflow_1_file_0.append(float(p[5]))
        elif p[4]=='File_7':
            workflow_1_file_1.append(float(p[5]))
        elif p[4]=='File_8':
            workflow_1_file_2.append(float(p[5]))
        elif p[4]=='File_9':
            workflow_1_file_3.append(float(p[5]))
        elif p[4]=='File_10':
            workflow_1_file_4.append(float(p[5]))
        else:
            workflow_1_file_5.append(float(p[5]))
    elif p[3]=='work_flow_2':
        if p[4]=='File_12':
            workflow_2_file_0.append(float(p[5]))
        elif p[4]=='File_13':
            workflow_2_file_1.append(float(p[5]))
        elif p[4]=='File_14':
            workflow_2_file_2.append(float(p[5]))
        elif p[4]=='File_15':
            workflow_2_file_3.append(float(p[5]))
        elif p[4]=='File_16':
            workflow_2_file_4.append(float(p[5]))
        else:
            workflow_1_file_5.append(float(p[5]))
    elif p[3]=='work_flow_3':
        if p[4]=='File_18':
            workflow_3_file_0.append(float(p[5]))
        elif p[4]=='File_19':
            workflow_3_file_1.append(float(p[5]))
        elif p[4]=='File_20':
            workflow_3_file_2.append(float(p[5]))
        elif p[4]=='File_21':
            workflow_3_file_3.append(float(p[5]))
        elif p[4]=='File_22':
            workflow_3_file_4.append(float(p[5]))
        else:
            workflow_3_file_5.append(float(p[5]))
    else:
        if p[4]=='File_24':
            workflow_4_file_0.append(float(p[5]))
        elif p[4]=='File_25':
            workflow_4_file_1.append(float(p[5]))
        elif p[4]=='File_26':
            workflow_4_file_2.append(float(p[5]))
        elif p[4]=='File_27':
            workflow_4_file_3.append(float(p[5]))
        elif p[4]=='File_28':
            workflow_4_file_4.append(float(p[5]))
        else:
            workflow_4_file_5.append(float(p[5]))

wf0_f0=np.array(workflow_0_file_1)
wf0_f1=np.array(workflow_0_file_2)
wf0_f2=np.array(workflow_0_file_3)
wf0_f3=np.array(workflow_0_file_4)
wf0_f4=np.array(workflow_0_file_5)
wf0_f5=np.array(workflow_0_file_5)

wf1_f0=np.array(workflow_1_file_0)
wf1_f1=np.array(workflow_1_file_1)
wf1_f2=np.array(workflow_1_file_2)
wf1_f3=np.array(workflow_1_file_3)
wf1_f4=np.array(workflow_1_file_4)
wf1_f5=np.array(workflow_1_file_5)

wf2_f0=np.array(workflow_2_file_0)
wf2_f1=np.array(workflow_2_file_1)
wf2_f2=np.array(workflow_2_file_2)
wf2_f3=np.array(workflow_2_file_3)
wf2_f4=np.array(workflow_2_file_4)
wf2_f5=np.array(workflow_2_file_5)

wf3_f0=np.array(workflow_3_file_0)
wf3_f1=np.array(workflow_3_file_1)
wf3_f2=np.array(workflow_3_file_2)
wf3_f3=np.array(workflow_3_file_3)
wf3_f4=np.array(workflow_3_file_4)
wf3_f5=np.array(workflow_3_file_5)

wf4_f0=np.array(workflow_4_file_0)
wf4_f1=np.array(workflow_4_file_1)
wf4_f2=np.array(workflow_4_file_2)
wf4_f3=np.array(workflow_4_file_3)
wf4_f4=np.array(workflow_4_file_4)
wf4_f5=np.array(workflow_4_file_5)


plt.figure()
plt.plot(wf0_f0, 'g')
plt.hold(True)
plt.plot(wf0_f1, 'r')
plt.plot(wf0_f2, 'b')
plt.plot(wf0_f3, 'y')
plt.plot(wf0_f4, 'c')
plt.plot(wf0_f5, 'm')
plt.axis([0,150,-0.1,1.1])
plt.grid(True)
plt.title('workflow 0')

plt.figure()
plt.plot(wf1_f0, 'g')
plt.hold(True)
plt.plot(wf1_f1, 'r')
plt.plot(wf1_f2, 'b')
plt.plot(wf1_f3, 'y')
plt.plot(wf1_f4, 'c')
plt.plot(wf1_f5, 'm')
plt.axis([0,150,-0.1,1.1])
plt.grid(True)
plt.title('workflow 1')

plt.figure()
plt.plot(wf2_f0, 'g')
plt.hold(True)
plt.plot(wf2_f1, 'r')
plt.plot(wf2_f2, 'b')
plt.plot(wf2_f3, 'y')
plt.plot(wf2_f4, 'c')
plt.plot(wf2_f5, 'm')
plt.axis([0,150,-0.1,1.1])
plt.grid(True)
plt.title('workflow 2')

plt.figure()
plt.plot(wf3_f0, 'g')
plt.hold(True)
plt.plot(wf3_f1, 'r')
plt.plot(wf3_f2, 'b')
plt.plot(wf3_f3, 'y')
plt.plot(wf3_f4, 'c')
plt.plot(wf3_f5, 'm')
plt.axis([0,150,-0.1,1.1])
plt.grid(True)
plt.title('workflow 3')

plt.figure()
plt.plot(wf4_f0, 'g')
plt.hold(True)
plt.plot(wf4_f1, 'r')
plt.plot(wf4_f2, 'b')
plt.plot(wf4_f3, 'y')
plt.plot(wf4_f4, 'c')
plt.plot(wf4_f5, 'm')
plt.axis([0,150,-0.1,1.1])
plt.grid(True)
plt.title('workflow 4')

plt.show()
