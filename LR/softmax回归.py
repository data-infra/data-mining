import matplotlib.pyplot as plt
import numpy as np
import random


# 样本数据集，第一列为x1，第二列为x2，第三列为分类（三种类别）
data=[
        [-2.68420713, 0.32660731, 0],[-2.71539062, -0.16955685, 0],[-2.88981954, -0.13734561, 0],[-2.7464372, -0.31112432, 0],[-2.72859298, 0.33392456, 0],
        [-2.27989736, 0.74778271, 0],[-2.82089068, -0.08210451, 0],[-2.62648199, 0.17040535, 0],[-2.88795857, -0.57079803, 0],[-2.67384469, -0.1066917, 0],
        [-2.50652679,0.65193501,0],[-2.61314272,0.02152063,0],[-2.78743398,-0.22774019,0],[-3.22520045,-0.50327991,0],[-2.64354322,1.1861949,0],
        [-2.38386932,1.34475434,0],[-2.6225262,0.81808967,0],[-2.64832273,0.31913667,0],[-2.19907796,0.87924409,0],[-2.58734619,0.52047364,0],
        [1.28479459, 0.68543919, 1],[0.93241075, 0.31919809, 1],[1.46406132, 0.50418983, 1],[0.18096721, -0.82560394, 1],[1.08713449, 0.07539039, 1],
        [0.64043675, -0.41732348, 1],[1.09522371, 0.28389121, 1],[-0.75146714, -1.00110751, 1],[1.04329778, 0.22895691, 1],[-0.01019007, -0.72057487, 1],
        [-0.5110862,-1.26249195,1],[0.51109806,-0.10228411,1],[0.26233576,-0.5478933,1],[0.98404455,-0.12436042,1],[-0.174864,-0.25181557,1],
        [0.92757294,0.46823621,1],[0.65959279,-0.35197629,1],[0.23454059,-0.33192183,1],[0.94236171,-0.54182226,1],[0.0432464,-0.58148945,1],
        [2.53172698, -0.01184224, 2],[1.41407223, -0.57492506, 2],[2.61648461, 0.34193529, 2],[1.97081495, -0.18112569, 2],[2.34975798, -0.04188255, 2],
        [3.39687992, 0.54716805, 2],[0.51938325, -1.19135169, 2],[2.9320051, 0.35237701, 2],[2.31967279, -0.24554817, 2],[2.91813423, 0.78038063, 2],
        [1.66193495,0.2420384,2],[1.80234045,-0.21615461,2],[2.16537886,0.21528028,2],[1.34459422,-0.77641543,2],[1.5852673,-0.53930705,2],
        [1.90474358,0.11881899,2],[1.94924878,0.04073026,2],[3.48876538,1.17154454,2],[3.79468686,0.25326557,2],[1.29832982,-0.76101394,2],
]


# 加载数据集，最后一列最为类别标签，前面的为特征属性的值
def loadDataSet(datasrc):
    # 生成X和y矩阵
    dataMat = np.mat(datasrc)
    y = dataMat[:, dataMat.shape[1] - 1]  # 最后一列为结果列
    b = np.ones(y.shape)  # 添加全1列向量代表b偏量
    X = np.column_stack((b, dataMat[:, 0:dataMat.shape[1] - 1]))  # 特征属性集和b偏量组成x
    X = np.mat(X)
    labeltype = np.unique(y.tolist())       # 获取分类数目
    eyes = np.eye(len(labeltype))    # 每一类用单位矩阵中对应的行代替，表示目标概率。如分类0的概率[1,0,0]，分类1的概率[0,1,0]，分类2的概率[0,0,1]
    Y=np.zeros((X.shape[0],len(labeltype)))
    for i in range(X.shape[0]):
        Y[i,:] = eyes[int(y[i,0])]               # 读取分类，替换成概率向量。这就要求分类为0,1,2,3,4,5这样的整数
    return X, y,Y       #X为特征数据集，y为分类数据集，Y为概率集



#可视化样本数据集
def plotDataSet():
    dataMat,labelMat,labelPMat = loadDataSet(data)                        #加载数据集
    plt.scatter(dataMat[:,1].flatten().A[0],dataMat[:,2].flatten().A[0],c=labelMat.flatten().A[0])                   #第一个偏量为b，第2个偏量x1，第3个偏量x2
    plt.xlabel('X1'); plt.ylabel('X2')                                 #绘制label
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


# softmax函数，将线性回归值转化为概率的激活函数。输入s要是行向量
def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=1)

# 逻辑回归中使用梯度下降法求回归系数。逻辑回归和线性回归中原理相同，只不过逻辑回归使用sigmoid作为迭代进化函数。因为逻辑回归是为了分类而生。线性回归为了预测而生
def gradAscent(dataMat, labelPMat):
    alpha = 0.2                                                        #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 1000                                                      #最大迭代次数
    weights = np.ones((dataMat.shape[1],labelPMat.shape[1]))             #初始化权回归系数矩阵  系数矩阵的行数为特征矩阵的列数，系数矩阵的列数为分类数目
    for k in range(maxCycles):
        h =  softmax(dataMat*weights)                                #梯度上升矢量化公式，计算预测值（行向量）。每一个样本产生一个概率行向量
        error = h-labelPMat                                            #计算每一个样本预测值误差
        weights = weights - alpha * dataMat.T * error                   # 根据所有的样本产生的误差调整回归系数
    return weights                                                     # 将矩阵转换为数组，返回回归系数数组


# 逻辑回归中使用随机梯度下降算法。numIter为迭代次数。改进之处：alpha移动步长是变化的。一次只用一个样本点去更新回归系数，这样就可以有效减少计算量了
def stocGradAscent(dataMatrix, labelPMat, numIter=500):
    m,n = np.shape(dataMatrix)                                                #返回dataMatrix的大小。m为样本对象的数目,n为列数。
    weights = np.ones((n,labelPMat.shape[1]))                                                  #参数初始化
    for j in range(numIter):
        for k in range(m):                                                    # 遍历m个样本对象
            alpha = 2/(1.0+j+k)+0.01                                          #降低alpha的大小，每次减小1/(j+i)。刚开始的时候可以步长大一点，后面调整越精细
            h = softmax(dataMatrix[k]*weights)                        #选择随机选取的一个样本，计算预测值h
            error = h-labelPMat[k]                              #计算一个样本的误差
            weights = weights - alpha * dataMatrix[k].T*error         #更新回归系数
    return weights.getA()                                                           #将矩阵转换为数组，返回回归系数数组

# 对新对象进行预测
def predict(weights,testdata):
    testdata.insert(0, 1.0)    #现在首部添加1代表b偏量
    testMat = np.mat([testdata])
    h=softmax(testMat * np.mat(weights))
    type = h.argmax(axis=1)   # 概率最大的分类就是预测分类
    return type,h   #type为所属分类，h为属于每种分类的概率

# 多分类只能绘制分界区域。而不是通过分割线来可视化
def plotBestFit(dataMat,labelMat,weights):

    # 获取数据边界值，也就属性的取值范围。
    x1_min, x1_max = dataMat[:, 1].min() - .5, dataMat[:, 1].max() + .5
    x2_min, x2_max = dataMat[:, 2].min() - .5, dataMat[:, 2].max() + .5
    # 产生x1和x2取值范围上的网格点，并预测每个网格点上的值。
    step = 0.02
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
    testMat = np.c_[xx1.ravel(), xx2.ravel()]   #形成测试特征数据集
    testMat = np.column_stack((np.ones(((testMat.shape[0]),1)),testMat))  #添加第一列为全1代表b偏量
    testMat = np.mat(testMat)
    # 预测网格点上的值
    y = softmax(testMat*weights)   #输出每个样本属于每个分类的概率
    # 判断所属的分类
    predicted = y.argmax(axis=1)                            #获取每行最大值的位置，位置索引就是分类
    predicted = predicted.reshape(xx1.shape).getA()
    # 绘制区域网格图
    plt.pcolormesh(xx1, xx2, predicted, cmap=plt.cm.Paired)

    # 再绘制一遍样本点，方便对比查看
    plt.scatter(dataMat[:, 1].flatten().A[0], dataMat[:, 2].flatten().A[0],
                c=labelMat.flatten().A[0],alpha=.5)  # 第一个偏量为b，第2个偏量x1，第3个偏量x2
    plt.show()






if __name__ == '__main__':

    # 查看数据集的分布
    # plotDataSet()

    dataMat, labelMat,labelPMat = loadDataSet(data)  # 加载数据集
    # dataMat = scale(dataMat)         # 特征属性去均值和归一化，如果使用归一化，则对新数据预测，也要进行归一化

    # 梯度下降法
    # weights = gradAscent(dataMat, labelPMat)         # 梯度下降法求回归系数
    weights = stocGradAscent(dataMat, labelPMat)    # 局部梯度下降法求回归系数
    print(weights)
    type,h = predict(weights,[1.90474358,0.11881899])
    print(type,h)
    plotBestFit(dataMat,labelMat,weights)