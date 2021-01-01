# 男女生身高分类
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xlrd

# svm 对于数据的要求: 所有的数据都要有label
# [155,48] -- 0 女生 [152,53] ---1  男生
# 监督学习 0 负样本 1 正样本

# 函数原型:vstack(tup) 参数tup可以是元组,列表,或者numpy数组,返回结果为numpy的数组
# 作用：它是垂直（按照行顺序）的把数组给堆叠起来。

# 1.准备数据

data_man=[] #先声明一个空list
data1 = xlrd.open_workbook("man.xlsx") #读取文件
table1 = data1.sheet_by_index(0) #按索引获取工作表，0就是工作表1
for i in range(table1.nrows): #table.nrows表示总行数
    line=table1.row_values(i) #读取每行数据，保存在line里面，line是list
    data_man.append(line) #将line加入到resArray中，resArray是二维list
man=np.array(data_man) #将resArray从二维list变成数组

data_woman=[] #先声明一个空list
data2 = xlrd.open_workbook("woman.xlsx") #读取文件
table2 = data2.sheet_by_index(0) #按索引获取工作表，0就是工作表1
for i in range(table2.nrows): #table.nrows表示总行数
    line=table2.row_values(i) #读取每行数据，保存在line里面，line是list
    data_woman.append(line) #将line加入到resArray中，resArray是二维list
woman=np.array(data_woman) #将resArray从二维list变成数组

data_test=[] #先声明一个空list
data3= xlrd.open_workbook("test.xlsx") #读取文件
table3 = data3.sheet_by_index(0) #按索引获取工作表，0就是工作表1
for i in range(table3.nrows): #table.nrows表示总行数
    line=table3.row_values(i) #读取每行数据，保存在line里面，line是list
    data_test.append(line) #将line加入到resArray中，resArray是二维list
test=np.array(data_test) #将resArray从二维list变成数组


# 2.data转换
# data = np.vstack(woman,man)                                       #报错
data = np.vstack((woman, man))  # 将两个训练样本用行堆积
data = np.array(data, dtype='float32')  # 改成float32类型
# 3.准备标签
#0-->女  1-->男
label=np.vstack(np.append(np.zeros([len(woman),1], dtype = int),np.ones([len(man),1], dtype = int)))

# 4.训练

# 创建svm
svm = cv2.ml.SVM_create()
# 设置svm属性
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)
# 训练
result = svm.train(data, cv2.ml.ROW_SAMPLE, label)

# 5.预测
pre_data = np.array(test, dtype='float32')
print("pre_data:\n", pre_data)
(par1, par2) = svm.predict(pre_data)  # 结果存储在par2中
print("par2:\n", par2)
