# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 创建单层决策树的数据集
def loadSimpData():
    datMat = np.array([
        [ 1. ,  2.1],
        [ 1.5,  1.6],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]
    ])
    classLabels =np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    return datMat,classLabels

# 加载数据集
def loadDataSet(fileName):
    alldata = np.loadtxt(fileName)
    n = alldata.shape[1]
    dataMat = alldata[:, 0:n-1]  # 添加数据
    labelMat = alldata[:, n-1]  # .astype(int).reshape(-1,1)  #添加标签
    return dataMat, labelMat

# 数据数据可视化
def showDataSet(dataMat, labelMat):
    # 绘制样本点
    place_plus = np.where(labelMat==1)[0]   # 正样本的位置
    place_minus = np.where(labelMat==-1)[0]  # 负样本的位置

    data_plus = dataMat[place_plus]    #正样本
    data_minus = dataMat[place_minus]  #负样本

    plt.scatter(np.transpose(data_plus)[0], np.transpose(data_plus)[1],s=30, alpha=0.7)   #正样本散点图
    plt.scatter(np.transpose(data_minus)[0], np.transpose(data_minus)[1], s=30, alpha=0.7) #负样本散点图
    plt.show()


# 就是根据数据集，要区分的特征，用来分类的特征的阈值进行计算分类结果。分类结果为1或-1分别表示两种类型
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    """
    单层决策树分类函数
    Parameters:
        dataMatrix - 数据矩阵
        dimen - 第dimen列，也就是第几个特征
        threshVal - 阈值
        threshIneq - 标志
    Returns:
        retArray - 分类结果
    """
    retArray = np.ones((np.shape(dataMatrix)[0],1))				#初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0	 	#如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0 		#如果大于阈值,则赋值为-1
    return retArray

# 获取决策树第一层分类信息。即获取最佳分类特征，以及该特征的分类阈值，和采用该阈值该特征进行分类的结果和误差
def buildStump(dataArr,classLabels,D):
    """
    找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    """
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
    minError = float('inf')														#最小误差初始化为正无穷大
    for i in range(n):															#遍历所有特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()		#找到特征中最小的值和最大值
        stepSize = (rangeMax - rangeMin) / numSteps								#计算步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:  										#大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize) 					#计算阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)#计算分类结果
                errArr = np.mat(np.ones((m,1))) 								#初始化误差矩阵
                errArr[predictedVals == labelMat] = 0 							#分类正确的,赋值为0
                weightedError = D.T * errArr  									#计算误差
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError: 									#找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    """
    使用AdaBoost算法提升弱分类器性能
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        numIt - 最大迭代次数
    Returns:
        weakClassArr - 训练好的分类器
        aggClassEst - 类别估计累计值
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)    										#初始化样本权重
    aggClassEst = np.mat(np.zeros((m,1)))                                   # 弱分类器的权重
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D) 	#构建单层决策树
        # print("D:",D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16))) 		#计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        bestStump['alpha'] = alpha  										#存储弱学习算法权重
        weakClassArr.append(bestStump)                  					#存储单层决策树
        # print("classEst: ", classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst) 	#计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()														#根据样本权重公式，更新样本权重
        #计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst  									#计算类别估计累计值
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1))) 	#计算误差
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate)
        if errorRate == 0.0: break 											#误差为0，退出循环
    return weakClassArr, aggClassEst


def adaClassify(datToClass,classifierArr):
    """
    AdaBoost分类函数
    Parameters:
        datToClass - 待分类样例
        classifierArr - 训练好的分类器
    Returns:
        分类结果
    """
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):										#遍历所有分类器，进行分类
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)

if __name__ == '__main__':
    # dataArr,classLabels = loadSimpData()
    # # showDataSet(dataArr,classLabels)
    # weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, classLabels)
    # print(adaClassify([[0,0],[5,5]], weakClassArr))

    dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')   # 加载训练集
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)  #AdaBoost算法形成
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')   # 加载测集
    print(weakClassArr)
    predictions = adaClassify(dataArr, weakClassArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))
    predictions = adaClassify(testArr, weakClassArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))
