import numpy as np
from sklearn.svm import SVC
import pandas as pd
import xlrd

dataman = pd.read_excel('man.xlsx', header = None)
# 导入数据集男性数据集((24, 9)）frame.DataFrame
dataman = np.array(dataman)
# 将DF数据框形式的数据转为ndarray形式数据
y_man = np.ones(dataman.shape[0])
# 构建男性标签列表形式24个
dataman_and_lable = np.column_stack((y_man,dataman))
# 将男性标签添加在男性数据的第一列，此时数据的shape为（（24，10））


datawoman = pd.read_csv("woman.csv", header = None)
# 同理读取女性数据集
datawoman = np.array(datawoman)
y_woman = np.zeros(datawoman.shape[0])
datawoman_and_lable = np.column_stack((y_woman,datawoman))
# datawoman的shape为（（23，10））

data_train_all = np.vstack((datawoman_and_lable, dataman_and_lable))  # 将两个训练样本用行堆积
# 此时data_train_all的shape为（（47，10））
data_train_all = np.array(data_train_all, dtype='float32')  # 改成float32类型


data_test=[] #先声明一个空list
data3= xlrd.open_workbook("test.xlsx") #读取文件
table3 = data3.sheet_by_index(0) #按索引获取工作表，0就是工作表1
for i in range(table3.nrows): #table.nrows表示总行数
    line=table3.row_values(i) #读取每行数据，保存在line里面，line是list
    data_test.append(line) #将line加入到resArray中，resArray是二维list
test=np.array(data_test) #将resArray从二维list变成数组


clf = SVC() # 此是SVM的离散多维分类算法函数，对于回归，用函数SVH

clf.fit(data_train_all[:,1:10], data_train_all[:,0]) # fit（特征值，标签值）函数用于训练模型
print('模型的训练集大小：')
print((len(dataman)+len(datawoman)))
print('测试数据集大小(按先男后女排列)：')
print(len(test))
print(test)
print('预测结果：（1-->  男    0-->  女）')
pre =  clf.predict(test)
print(pre) # 打印函数的预测输出

num = 0 # 检测成功率（测试集按 男女男女排列）
for i in range(len(test)):
    if pre[i]==1 :
        num+=1
print('预测成功率：')
print(1-abs(num-len(test)/2)/(len(test)/2))



































