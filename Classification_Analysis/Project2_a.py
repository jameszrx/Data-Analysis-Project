from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import numpy as np

#load the training data
cate = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
datasets = fetch_20newsgroups(subset = 'train', categories = cate, shuffle = True, random_state = 42, remove = ('headers','footers','quotes'))

#sorting the data into 20 different groups, save them in the "data" list and get the length of each group
index = list()
length = list()
data = list()
for j in range(20):
    index_temp = list()
    index_temp.append(list(np.where(datasets.target == j))[0])
    index.append(index_temp)
    data_temp = list()
    for i in index[j][0]:
        data_temp.append(datasets.data[i])
    data.append(data_temp)
    length.append(len(data_temp))

#plot the histogram
plt.figure()
index_plt = range(20)
width = 1
c = ['b', 'r', 'r', 'r', 'r', 'b', 'b', 'g', 'g', 'g', 'g', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']
p = plt.bar(index_plt, length, width, color = c)
label_index = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
plt.xticks(label_index, ('athe', 'grap', 'misc', 'ibm', 'mac', 'win', 'fors', 'autos', 'motor', 'base', 'hock', 'crypt', 'elec', 'med', 'space', 'chris', 'guns', 'mid', 'p-misc', 'r-misc'))
plt.ylim([350, 610])
plt.ylabel('number of documents per topic')
plt.legend((p[1], p[7]), ('Computer Technology', 'Recreational Activity'), loc = 'upper right')
len_recreational = length[7] + length[8] + length[9] + length[10]
len_computer = length[1] + length[2] + length[3] + length[4]
plt.title('Document Numbers:    Recreational: %s, Computer: %s' % (str(len_recreational), str(len_computer)))
plt.grid(True)
plt.show()


