# 构建一个葡萄酒样本数据集。等级、年代、价格
from random import random,randint
import math

# 根据等级和年代对价格进行模拟
def wineprice(rating,age):
    peak_age=rating-50

    # 根据等级计算价格
    price=rating/2
    if age>peak_age:
        # 经过“峰值年”，后续5年里其品质将会变差
        price=price*(5-(age-peak_age)/2)
    else:
        # 价格在接近“峰值年”时会增加到原值的5倍
        price=price*(5*((age+1)/peak_age))
    if price<0: price=0
    return price

# 生成一批数据代表样本数据集
def wineset1():
    rows=[]
    for i in range(300):
        # 随机生成年代和等级
        rating=random()*50+50
        age=random()*50

        # 得到一个参考价格
        price=wineprice(rating,age)

        # 添加一些噪音
        price*=(random()*0.2+0.9)

        # 加入数据集
        rows.append({'input':(rating,age),'result':price})
    return rows

# 使用欧几里得距离，定义距离
def euclidean(v1,v2):
    d=0.0
    for i in range(len(v1)):
        d+=(v1[i]-v2[i])**2
    return math.sqrt(d)

# 计算待测商品和样本数据集中任一商品间的距离。data样本数据集，vec1待测商品
def getdistances(data,vec1):
    distancelist=[]

    # 遍历样本数据集中的每一项
    for i in range(len(data)):
        vec2=data[i]['input']

        # 添加距离到距离列表
        distancelist.append((euclidean(vec1,vec2),i))

    # 距离排序
    distancelist.sort()
    return distancelist  #返回距离列表

# 对距离值最小的前k个结果求平均
def knnestimate(data,vec1,k=5):
    # 得到经过排序的距离值
    dlist=getdistances(data,vec1)
    avg=0.0

    # 对前k项结果求平均
    for i in range(k):
        idx=dlist[i][1]
        avg+=data[idx]['result']
    avg=avg/k
    return avg


#=================上面是普通的k最近邻算法==============
# ================下面是带权值的k最近邻===============

# 使用反函数为近邻分配权重
def inverseweight(dist,num=1.0,const=0.1):
    return num/(dist+const)

# 使用减法函数为近邻分配权重。当最近距离都大于const时不可用
def subtractweight(dist,const=1.0):
    if dist>const:
        return 0
    else:
        return const-dist

# 使用高斯函数为距离分配权重
def gaussian(dist,sigma=5.0):
    return math.e**(-dist**2/(2*sigma**2))


# 对距离值最小的前k个结果求加权平均
def weightedknn(data,vec1,k=5,weightf=gaussian):
    # 得到距离值
    dlist=getdistances(data,vec1)
    avg=0.0
    totalweight=0.0

    # 得到加权平均
    for i in range(k):
        dist=dlist[i][0]
        idx=dlist[i][1]
        weight=weightf(dist)
        avg+=weight*data[idx]['result']
        totalweight+=weight
    if totalweight==0: return 0
    avg=avg/totalweight
    return avg



# ========================交叉验证====================
# 验证你的算法好坏，用来挑选不同的算法或算法参数。

# 划分数据。test待测集占的比例。其他数据为训练集
def dividedata(data,test=0.05):
    trainset=[]
    testset=[]
    for row in data:
        if random()<test:
            testset.append(row)
        else:
            trainset.append(row)
    return trainset,testset


# 对使用算法进行预测的结果的误差进行统计，以此判断算法好坏。algf为算法函数，trainset为训练数据集，testset为待测数据集
def testalgorithm(algf,trainset,testset):
    error=0.0
    for row in testset:
        guess=algf(trainset,row['input'])   #这一步要和样本数据的格式保持一致，列表内个元素为一个字典，每个字典包含input和result两个属性。而且函数参数是列表和元组
        error+=(row['result']-guess)**2
        #print row['result'],guess
    #print error/len(testset)
    return error/len(testset)

# 将数据拆分和误差统计合并在一起。对数据集进行多次划分，并验证算法好坏
def crossvalidate(algf,data,trials=100,test=0.1):
    error=0.0
    for i in range(trials):
        trainset,testset=dividedata(data,test)
        error+=testalgorithm(algf,trainset,testset)
    return error/trials


# =====================不同类型的变量、变量的不同值域、变量无效的情况=================

# 构建新数据集，模拟不同类型、尺度的属性
def wineset2():
    rows=[]
    for i in range(300):
        rating=random()*50+50   #酒的档次
        age=random()*50         #酒的年限
        aisle=float(randint(1,20))  #酒箱的号码（无关属性）
        bottlesize=[375.0,750.0,1500.0][randint(0,2)]  #酒的容量
        price=wineprice(rating,age)  #酒的价格
        price*=(bottlesize/750)
        price*=(random()*0.2+0.9)
        rows.append({'input':(rating,age,aisle,bottlesize),'result':price})
    return rows

# 按比例对属性进行缩放，scale为各属性的值的缩放比例。
def rescale(data,scale):
    scaleddata=[]
    for row in data:
        scaled=[scale[i]*row['input'][i] for i in range(len(scale))]
        scaleddata.append({'input':scaled,'result':row['result']})
    return scaleddata

# 既然要缩放，究竟各属性缩放多少才能既保证属性的重要程度，又不至于降低其他属性的缩放程度。使用优化技术来获取每个属性的缩放比例
# 生成成本函数。闭包
def createcostfunction(algf,data):
    def costf(scale):
        sdata=rescale(data,scale)
        return crossvalidate(algf,sdata,trials=10)
    return costf

weightdomain=[(0,10)]*4     #将缩放比例这个题解的取值范围设置为0-10，可以自己设定，用于优化算法


# 对于样本数据集包含多种分布情况时，输出结果我们也希望不唯一。我们可以使用概率结果进行表示，输出每种结果的值和出现的概率
# 构造新数据集。酒有可能是从逃税酒，而样本数据集中没有记录这一特性
def wineset3():
    rows=wineset1()
    for row in rows:
        if random()<0.5:
            # 一半的可能是逃税酒
            row['result']*=0.6
    return rows

# 计算概率。data样本数据集，vec1预测数据，low，high结果范围，weightf为根据距离进行权值分配的函数
def probguess(data,vec1,low,high,k=5,weightf=gaussian):
    dlist=getdistances(data,vec1)  #获取距离列表
    nweight=0.0
    tweight=0.0

    for i in range(k):
        dist=dlist[i][0]   #距离
        idx=dlist[i][1]   #索引号
        weight=weightf(dist)  #权值
        v=data[idx]['result']  #真实结果

        # 当前数据点位于指定范围内么？
        if v>=low and v<=high:
            nweight+=weight    #指定范围的权值之和
        tweight+=weight        #总的权值之和
    if tweight==0: return 0

    # 概率等于位于指定范围内的权重值除以所有权重值
    return nweight/tweight

from pylab import *

# 绘制累积概率分布图。data样本数据集，vec1预测数据，high取值最高点，k近邻范围，weightf权值分配
def cumulativegraph(data,vec1,high,k=5,weightf=gaussian):
    t1=arange(0.0,high,0.1)
    cprob=array([probguess(data,vec1,0,v,k,weightf) for v in t1])   #预测产生的不同结果的概率
    plot(t1,cprob)
    show()

# 绘制概率密度图
def probabilitygraph(data,vec1,high,k=5,weightf=gaussian,ss=5.0):
    # 建立一个代表价格的值域范围
    t1=arange(0.0,high,0.1)

    # 得到整个值域范围内的所有概率
    probs=[probguess(data,vec1,v,v+0.1,k,weightf) for v in t1]

    # 通过加上近邻概率的高斯计算结果，对概率值做平滑处理
    smoothed=[]
    for i in range(len(probs)):
        sv=0.0
        for j in range(0,len(probs)):
            dist=abs(i-j)*0.1
            weight=gaussian(dist,sigma=ss)
            sv+=weight*probs[j]
        smoothed.append(sv)
    smoothed=array(smoothed)

    plot(t1,smoothed)
    show()



if __name__=='__main__':  #只有在执行当前模块时才会运行此函数
    # data = wineset1()      #创建第1批数据集
    # print(data)

    #=======================预测=======================
    # result=knnestimate(data,(95.0,3.0))   #根据最近邻直接求平均进行预测
    # print(result)
    # result=weightedknn(data,(95.0,3.0),weightf=inverseweight)   #使用反函数做权值分配方法，进行加权平均
    # print(result)
    # result = weightedknn(data, (95.0, 3.0), weightf=subtractweight)  # 使用减法函数做权值分配方法，进行加权平均
    # print(result)
    # result = weightedknn(data, (95.0, 3.0), weightf=gaussian)  # 使用高斯函数做权值分配方法，进行加权平均
    # print(result)


    #=======================交叉验证========================
    # error = crossvalidate(knnestimate,data)   #对直接求均值算法进行评估
    # print('平均误差：'+str(error))
    #
    # def knn3(d,v): return knnestimate(d,v,k=3)  #定义一个函数指针。参数为d列表，v元组
    # error = crossvalidate(knn3, data)            #对直接求均值算法进行评估
    # print('平均误差：' + str(error))
    #
    # def knninverse(d,v): return weightedknn(d,v,weightf=inverseweight)  #定义一个函数指针。参数为d列表，v元组
    # error = crossvalidate(knninverse, data)            #对使用反函数做权值分配方法进行评估
    # print('平均误差：' + str(error))


    # ======================缩放比例优化======================
    # data = wineset2()  # 创建第2批数据集
    # print(data)
    # import optimization
    # costf=createcostfunction(knnestimate,data)      #创建成本函数
    # result = optimization.annealingoptimize(weightdomain,costf,step=2)    #使用退火算法寻找最优解
    # print(result)

    # ======================不对称分布，产生结果不唯一，每种结果具有各自的概率======================
    data = wineset3()  # 创建第3批数据集
    print(data)
    cumulativegraph(data,(1,1),120)   #绘制累积概率密度
    probabilitygraph(data,(1,1),6)    #绘制概率密度图







