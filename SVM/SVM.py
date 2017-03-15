import numpy as np
import random
from numpy import arange
from sklearn import metrics
from sklearn.utils import shuffle
import pickle
import gzip
from sklearn import svm
import time
import pdb
from math import *

# # 实现SVM算法最重要方法，序列最小优化。本代码实现了根据《统计学习方法》-李航 
# #以及 《机器学习实战》给出了简化版本的SMO算法实现 
# def SMO(dataMatIn,classLabels,C=0.6, tol=0.0001, maxIter=1000, maxNum):
# # SMO输入参数为：输入数据，标记，常数C，容错率，取消前最大迭代次数,
# # 防止SVM收敛太慢提前退出的阈值
#     # 创建一个alpha向量并将其初始化为0向量
#     #     当迭代次数小于最大迭代次数时（外循环）：
#     #         对数据集中的每个数据向量（内循环）：
#     #             如果该数据向量可以被优化：（违反KKT条件最严重那一个）
#     #                 随机选择另一个数据向量（有约束条件自动确定）
#     #                 同时优化这两个向量
#     #                 如果两个向量都不能被优化，退出内循环
#     #         如果所有向量都没有被优化，增加迭代数目，进行下一轮循环

# #该算法给出了多类SVM实现的方法，主要思想是一一区分法：构造k(k-1)/2个分类器，
# #即每两类之间构成一个SVM，累计各类别的得分, 选择得分最高者所对应的类别为测试数据的类别。
# #因此本文中共为10个类别构造了45个SVM二类分类器，这也是运用本文算法进行预测耗费时间巨大的主要原因
# def Multi_SVM(dataMatIn, classLabels, C=1, tol=0.0001, maxIter=1000,maxNum):
#     #将数据按照0,1,2...9类别进行分割
#     splitsourceData(dataMatIn,classLabels)
#     resultW = []
#     resultb = []
#     count = 0
#     #构建10*9/2 = 45个SVM二类分类器
#     for i in range(10):
#         for j in range(i+1,10):
#             print("%dth iteration" %count)
#             #类别两两之间构建一个SVM分类器
#             b,alphas = SMO(np.mat(Data),np.mat(Label),maxIter=maxIter,maxNum=maxNum)
#             w = calcW(alphas,np.mat(tmpData),np.mat(changeLabel))
#             resultW.append([i,j,w])
#             count += 1
#             resultb.append(b)
#     #根据计算好的b,w值，预测样本类别，具体计算如下：
#     对于每一个测试样本用45分类器分类并记录分到各个类别的
#     总次数，选择总次数最大的作为该样本的分类类别，如果出现两个或者两个以上
#     类别一样的则随机选择一个类别作为该样本的分类类别



#用于SMO随机随着一个不同于当前点的点
def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))  # 一直在挑选随机数j，直到不等于i，随机数的范围在0~m
    return j  # 返回挑选好的随机数

# 限定范围，最大不能超过H，最小不能低于
def clipAlpha(aj,H,L):  L
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

# 实现SVM算法最重要方法，序列最小优化。本代码实现了根据《统计学习方法》-李航 以及 《机器学习实战》
# 给出了简化版本的SMO算法实现 
#SMO输入参数为：输入数据，标记，常数C，容错率，取消前最大迭代次数,防止SVM收敛太慢提前退出的阈值
def SMO(dataMatIn, classLabels, C=0.6, tol=0.0001, maxIter=1000, maxNum):
    dataMatrix = np.mat(dataMatIn);   # 转换成矩阵
    labelMat = np.mat(classLabels).transpose()  # 转换成矩阵，并转置，标记成为一个列向量，每一行和数据矩阵对应
    m,n = dataMatrix.shape  # 行，列    

    b = 0;  # 参数b的初始化
    alphas = np.mat(np.zeros((m,1)))  # 参数alphas是个list，初始化也是全0，大小等于样本数
    iter = 0  # 当前迭代次数，maxIter是最大迭代次数
    SMO_start_time = time.time()
    print(len(dataMatIn),len(classLabels))
    for kk in range(maxNum):  # 防止SVM收敛太慢提前退出的阈值

        alphaPairsChanged = 0  # 标记位，记录alpha在该次循环中，有没有优化
        for i in range(m):  # 第i个样本
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b  # 第i样本的预测类别
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions  # 误差
            #是否可以继续优化
            if ((labelMat[i]*Ei < -tol) and (alphas[i] < C)) or ((labelMat[i]*Ei > tol) and (alphas[i] > 0)):
                j = selectJrand(i,m)  # 随机选择第j个样本
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b  # 样本j的预测类别
                Ej = fXj - float(labelMat[j])  # 误差

                alphaIold = alphas[i].copy();  # 拷贝，分配新的内存
                alphaJold = alphas[j].copy();
                #保证alpha在0~C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L==H: 
                    continue

                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T

                if eta >= 0: 

                    continue

                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)  # 门限函数阻止alpha_j的修改量过大

                #如果修改量很微小
                if (abs(alphas[j] - alphaJold) < 0.00001)
                    continue

                # 更新alpha_i，其中alpha_i的修改方向相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                # 为两个alpha设置常数项b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0

                # 说明alpha已经发生改变
                alphaPairsChanged += 1

        #如果没有更新，那么继续迭代；如果有更新，那么迭代次数归0，继续优化
        if (alphaPairsChanged == 0): 
            iter += 1
        else: 
            iter = 0
        #如果检测过程中发现没有更新的迭代次数已经超过取消前给定的最大迭代次数，则break
        if iter > maxIter:
            break
            print("break")
        kk += 1
    #记录每一轮SMO算法处理时间
    SMO_end_time = time.time()
    print ("SMO once time:", SMO_end_time - SMO_start_time)
    return b,alphas

    #计算W值，用于之后预测计算
def calcW(alphas,dataArr,classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m,n = X.shape
    w = np.zeros((n,1))
    for i in range(m):
        ab = np.multiply(alphas[i]*labelMat[i],X[i,:].T)
        w += ab
    return w

#将原始数据按照0,1,2...,9的格式进行分割
def splitsourceData(dataIn,labelIn):
    labels = [i for i in range(10)]
    splitDataOut = [[],[],[],[],[],[],[],[],[],[]]
    splitLabelOut = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(labelIn)):
        for j in range(len(labels)):
            if labelIn[i]==labels[j]:
                splitDataOut[labels[j]].append(dataIn[i])
                splitLabelOut[labels[j]].append(labelIn[i])
    return splitDataOut,splitLabelOut

#对于每一个二类的SMO进行预测输出
def predict(testMatrix,b,resultW):
    testMat = np.mat(testMatrix)
    fianlresult = []
    for i in range(testMat.shape[0]):
        eachSVMresult = [0 for i in range(10)]
        for j in range(len(resultW)):
            if testMat[i]*resultW[j][2]+b[j]>0:
                eachSVMresult[resultW[j][0]] += 1
            else:
                 eachSVMresult[resultW[j][1]] += 1
        fianlresult.append(eachSVMresult.index(max(eachSVMresult)))
    # return[1 if testMat[i]*w+b[i]>0 else -1 for i in range(testMat.shape[0])]
    return fianlresult

#该算法给出了多类SVM实现的方法，主要思想是一一区分法：构造k(k-1)/2个分类器，
#即每两类之间构成一个SVM，累计各类别的得分, 选择得分最高者所对应的类别为测试数据的类别。
#因此本文中共为10个类别构造了45个SVM二类分类器，这也是运用本文算法进行预测耗费时间巨大的主要原因
def Multi_SVM(dataMatIn, classLabels, C=1, tol=0.0001, maxIter=1000,maxNum):
    splitData,splitLabel = splitsourceData(dataMatIn,classLabels)
    resultW = []
    resultb = []
    count = 0
    for i in range(10):
        # print("i= %d" %i)
        for j in range(i+1,10):
            print("%dth iteration" %count)
            tmpData = splitData[i]+splitData[j]
            tmpLabel = splitLabel[i]+splitLabel[j]
            if splitLabel[i]==None or splitLabel[j]==None:
                continue
            # print(splitLabel[i],splitLabel[j])
            # print("tmpLabel")
            # print(tmpLabel)
            changeLabel = [1 if tmpLabel[k]==i else -1 for k in range(len(tmpLabel))]
            b,alphas = SMO(np.mat(tmpData),np.mat(changeLabel),maxIter=maxIter,maxNum=maxNum)
            w = calcW(alphas,np.mat(tmpData),np.mat(changeLabel))
            resultW.append([i,j,w])
            # print(i,j)
            count += 1
            # print(count)
            resultb.append(b)
    return resultW,resultb

#用于导入mnist数据集，主要分为50000训练集，10000验证集，10000测试集
def mnist_load():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin-1')
    f.close()
    return (training_data, validation_data, test_data)

#该方法直接调用了sklearn中的SVM算法进行mnist数据的分类
def run():
    training_data, validation_data, test_data = mnist_load()
    X_train = training_data[0]
    Y_train = training_data[1]
    X_test = test_data[0]
    Y_test = test_data[1]
    # pdb.set_trace()
    # train
    print("SVM begin:")
    start = time.time()
    clf = svm.SVC()
    clf.fit(X_train, Y_train)
    end = time.time()
    print("SVM end:")
    print("SVM overall time:", end - start)
    # test
    predictions = [int(a) for a in clf.predict(X_test)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, Y_test))
    print ("Baseline classifier using an SVM.")
    print ("%s of %s values correct." % (num_correct, len(Y_test)))

    # Evaluate the prediction
    print ("Evaluating results...")
    print ("Precision: \t", metrics.precision_score(Y_test, predictions,average="macro"))
    print ("Recall: \t", metrics.recall_score(Y_test, predictions,average="macro"))
    print ("F1 score: \t", metrics.f1_score(Y_test, predictions,average="macro"))
    # print ("Mean accuracy: \t", clf.score(X_test, Y_test))

if __name__ == "__main__":
    rootPath = "E:/OneDrive/机器学习原理/SVM作业"
    save_resultW = open(rootPath+"/resultW.txt", 'w')
    save_resultb = open(rootPath+"/resultb.txt", 'w')
    load_start_time = time.time()

    training_data, validation_data, test_data = mnist_load()
    a = random.sample(range(50000),15000)
    X_train = [training_data[0][a[i]] for i in range(15000)]
    Y_train = [training_data[1][a[i]] for i in range(15000)]
    print(len(X_train))
    X_test = []
    Y_test = []

    X_test.append(test_data[0][0:10])
    Y_test.append(test_data[1][0:10])
    X_test.append(test_data[0][10:60])
    Y_test.append(test_data[1][10:60])
    X_test.append(test_data[0][60:160])
    Y_test.append(test_data[1][60:160])
    X_test.append(test_data[0][160:460])
    Y_test.append(test_data[1][160:460])
    X_test.append(test_data[0][460:960])
    Y_test.append(test_data[1][460:960])
    X_test.append(test_data[0][1000:2000])
    Y_test.append(test_data[1][1000:2000])
    X_test.append(test_data[0][2000:7000])
    Y_test.append(test_data[1][2000:7000])
    X_test.append(test_data[0][0:8000])
    Y_test.append(test_data[1][0:8000])
    X_test.append(test_data[0])
    Y_test.append(test_data[1])
    load_end_time = time.time()
    print ("load time:",load_end_time - load_start_time)

    SVM_start_time = time.time()
    resultW,resultb = Multi_SVM(X_train,Y_train,maxIter=4,maxNum=500)
    # save_resultb.write(str(resultb[i]) for i in range(len(resultW)))
    # # pdb.set_trace()
    # # save_resultb.write(resultb[i] for i in range(len(resultW)))   
    # save_resultW.write(resultb)
    SVM_end_time = time.time()
    print ("SVM time:",SVM_end_time - SVM_start_time)
    count = 0
    while count <len(X_test):
        testMatrix = np.mat(X_test[count])
        finalresult = predict(testMatrix, resultb, resultW)

        # test
        # predictions = [int(a) for a in clf.predict(X_test)]
        num_correct = sum(int(a == y) for a, y in zip(finalresult, Y_test[count]))
        print ("Baseline classifier using an SVM.")
        print ("%s of %s values correct." % (num_correct, len(Y_test[count])))
        precision = float(num_correct/len(Y_test[count]))
        print("Precision:\t %f" %precision)
        # pdb.set_trace()
        count += 1

    # b,alphas = SMO(dataMatrix,labelMat,maxIter=2)
    # testMatrix = np.mat([[2,3,-1,10],[26,2,4,6]])
    # w = np.mat(calcW(alphas,dataMatrix,labelMat))
    # pre = predict(testMatrix,b,w)
    # print(pre)
    # start_time = time.time()
    # results = run()
    # end_time = time.time()
    # print ("Overall running time:", end_time - start_time)