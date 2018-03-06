import matplotlib.pyplot as plt
import numpy as np
import random


# 样本数据集，第一列为x1，第二列为x2，第三列为分类（二种类别）
data=[
    [-0.017612, 14.053064, 0],[-0.752157, 6.538620, 0],[-1.322371, 7.152853, 0],[0.423363, 11.054677, 0],[0.569411, 9.548755, 0],
    [-0.026632, 10.427743, 0],[0.667394, 12.741452, 0],[1.347183, 13.175500, 0],[1.217916, 9.597015, 0],[-0.733928, 9.098687, 0],
    [1.416614, 9.619232, 0],[1.388610, 9.341997, 0],[0.317029, 14.739025, 0],[-0.576525, 11.778922, 0],[-1.781871, 9.097953, 0],
    [-1.395634, 4.662541, 1],[0.406704, 7.067335, 1],[-2.460150, 2.866805, 1],[0.850433, 6.920334, 1],[1.176813, 3.167020, 1],
    [-0.566606, 5.749003, 1],[0.931635, 1.589505, 1],[-0.024205, 6.151823, 1],[-0.036453, 2.690988, 1],[-0.196949, 0.444165, 1],
    [1.014459, 5.754399, 1],[1.985298, 3.230619, 1],[-1.693453, -0.557540, 1],[-0.346811, -1.678730, 1],[-2.124484, 2.672471, 1]
]


#加载数据集，最后一列最为类别标签，前面的为特征属性的值
def loadDataSet(datasrc):
    dataMat = np.mat(datasrc)
    y = dataMat[:, dataMat.shape[1] - 1]  # 最后一列为结果列
    b = np.ones(y.shape)  # 添加全1列向量代表b偏量
    X = np.column_stack((b, dataMat[:, 0:dataMat.shape[1] - 1]))  # 特征属性集和b偏量组成x
    X = np.mat(X)
    labeltype = np.unique(y.tolist())  # 获取分类数目
    eyes = np.eye(len(labeltype))  # 每一类用单位矩阵中对应的行代替，表示目标概率。如分类0的概率[1,0,0]，分类1的概率[0,1,0]，分类2的概率[0,0,1]
    Y = np.zeros((X.shape[0], len(labeltype)))
    for i in range(X.shape[0]):
        Y[i, :] = eyes[int(y[i, 0])]  # 读取分类，替换成概率向量。这就要求分类为0,1,2,3,4,5这样的整数
    # print(Y)
    return X, y, Y  # X为特征数据集，y为分类数据集，Y为概率集



#可视化样本数据集
def plotDataSet():
    dataMat,labelMat,labelPMat = loadDataSet(data)                        #加载数据集
    plt.scatter(dataMat[:,1].flatten().A[0],dataMat[:,2].flatten().A[0],c=labelMat.flatten().A[0])                   #第一个偏量为b，第2个偏量x1，第3个偏量x2
    plt.xlabel('X1'); plt.ylabel('X2')                                 #绘制label
    plt.xlim([-3,3])
    plt.ylim([-4,16])
    plt.show()


# 对特征数据集去均值和归一化，收敛速度和收敛效果更佳
def scale(dataMat):
    saledMat = dataMat.copy()
    for i in range(dataMat.shape[1]):    #每列分别归一化
        onecolumn = dataMat[:,i]
        onestd = onecolumn.std()
        if onestd==0:saledcolumn = onecolumn
        else:saledcolumn = (onecolumn-onecolumn.mean())/onestd
        saledMat[:,i] = saledcolumn
    return saledMat

# sigmoid函数，逻辑回归函数，将线性回归值转化为概率的激活函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

# 逻辑回归中使用梯度下降法求回归系数。逻辑回归和线性回归中原理相同，只不过逻辑回归使用sigmoid作为迭代进化函数。因为逻辑回归是为了分类而生。线性回归为了预测而生
def gradAscent(dataMat, labelPMat):
    m, n = np.shape(dataMat)                                            #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.05                                                        #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 1000                                                      #最大迭代次数
    weights = np.ones((n,labelPMat.shape[1]))                            #初始化权重列向量
    for k in range(maxCycles):
        h =  sigmoid(dataMat * weights)                                #梯度上升矢量化公式，计算预测值（列向量）。每一个样本产生一个预测值
        error = h-labelPMat                                            #计算每一个样本预测值误差
        weights = weights - alpha * dataMat.T * error                   # 根据所有的样本产生的误差调整回归系数
    return weights.getA()                                               # 将矩阵转换为数组，返回回归系数数组

# 逻辑回归中使用随机梯度下降算法。numIter为迭代次数。改进之处：alpha移动步长是变化的。一次只用一个样本点去更新回归系数，这样就可以有效减少计算量了
def stocGradAscent(dataMatrix, labelPMat, numIter=500):
    m,n = np.shape(dataMatrix)                                                #返回dataMatrix的大小。m为样本对象的数目,n为列数。
    weights = np.ones((n,labelPMat.shape[1]))                                                  #参数初始化
    for j in range(numIter):
        for k in range(m):                                                    # 遍历m个样本对象
            alpha = 10/(1.0+j+k)+0.01                                          #降低alpha的大小，每次减小1/(j+i)。刚开始的时候可以步长大一点，后面调整越精细
            h = sigmoid(dataMatrix[k]*weights)                        #选择随机选取的一个样本，计算预测值h
            error = h-labelPMat[k]                              #计算一个样本的误差
            weights = weights - alpha * dataMatrix[k].T*error         #更新回归系数
    return weights.getA()                                                           #将矩阵转换为数组，返回回归系数数组



# 对新对象进行预测
def predict(weights,testdata):
    testdata.insert(0, 1.0)       #现在首部添加1代表b偏量
    testMat = np.mat([testdata])
    y=sigmoid(testMat * np.mat(weights))  # 对输入进行预测
    # print('输出结果：',y)
    type = y.argmax(axis=1)     # 概率最大的分类就是预测分类。由于输出值y为行向量，所按行取最大值的位置
    return type, y  # type为所属分类，h为属于每种分类的概率

# 分类只能绘制分界区域。而不是通过分割线来可视化
def plotBestFit(dataMat,labelMat,weights):

    # 先产生x1和x2取值范围上的网格点，并预测每个网格点上的值。
    step = 0.02
    x1_min, x1_max = -3,3
    x2_min, x2_max = -4,16
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
    testMat = np.c_[xx1.ravel(), xx2.ravel()]   #形成测试特征数据集
    testMat = np.column_stack((np.ones(((testMat.shape[0]),1)),testMat))  #添加第一列为全1代表b偏量
    testMat = np.mat(testMat)
    y = sigmoid(testMat*weights)   #输出每个样本属于每个分类的概率
    predicted = y.argmax(axis=1)                            #获取每行最大值的位置，位置索引就是分类
    predicted = predicted.reshape(xx1.shape).getA()
    # 绘制区域网格图
    plt.pcolormesh(xx1, xx2, predicted, cmap=plt.cm.Paired)

    # 再绘制一遍样本数据点，这样方便查看
    plt.scatter(dataMat[:, 1].flatten().A[0], dataMat[:, 2].flatten().A[0],
                c=labelMat.flatten().A[0], alpha=.5)  # 第一个偏量为b，第2个偏量x1，第3个偏量x2

    plt.xlim([-3,3])
    plt.ylim([-4,16])
    plt.show()

if __name__ == '__main__':
    # 查看数据集的分布
    # plotDataSet()


    dataMat, labelMat, labelPMat = loadDataSet(data)   # 加载数据集
    # print(list(zip(dataMat,labelMat)))
    # dataMat = scale(dataMat)   # 特征属性去均值和归一化，如果使用归一化，则对新数据预测，也要进行归一化
    # weights = gradAscent(dataMat, labelPMat)         # 梯度下降法求回归系数
    weights = stocGradAscent(dataMat, labelPMat)    # 局部梯度下降法求回归系数
    print(weights)
    type,h = predict(weights,[0.317029,14.739025])
    print(type,h)
    plotBestFit(dataMat,labelMat,weights)