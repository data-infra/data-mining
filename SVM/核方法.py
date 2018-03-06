# 输入对象的各个属性间存在非线性作用
# 使用核方法，进行婚姻数据匹配。婚姻数据同线性分类器中数据

# 场景：男女不同的属性信息，例如年龄、是否吸烟、是否要孩子、兴趣列表、家庭住址。产生的输出结果，配对成功还是不成功

import linearclassify

#=============核方法============
import math

# 向量线段长度。
def veclength(v):
    return sum([p**2 for p in v])

# 使用径向基函数代替向量点积函数。将数据映射到更高维的空间（以为更高纬度空间可以通过线性分离）。可以调整gamma参数，达到最佳分离
def rbf(v1,v2,gamma=10):
    dv=[v1[i]-v2[i] for i in range(len(v1))]
    l=veclength(dv)
    return math.e**(-gamma*l)

# 使用核方法进行线性分类。计算每个坐标点与分类中其余每个坐标点之间的点积或径向基函数的结果，然后对他们求均值
def nlclassify(point,rows,offset,gamma=10):
    sum0=0.0
    sum1=0.0
    count0=0
    count1=0

    for row in rows:
        if row.matchresult==0:
            sum0+=rbf(point,row.data,gamma)  #求径向基函数
            count0+=1
        else:
            sum1+=rbf(point,row.data,gamma)
            count1+=1
    y=(1.0/count0)*sum0-(1.0/count1)*sum1+offset

    if y>0: return 0
    else: return 1

def getoffset(rows,gamma=10):
    t0=[]
    t1=[]
    for row in rows:
        if row.matchresult==0: t0.append(row.data)
        else: t1.append(row.data)
    sum0=sum(sum([rbf(v1,v2,gamma) for v1 in t0]) for v2 in t0)
    sum1=sum(sum([rbf(v1,v2,gamma) for v1 in t1]) for v2 in t1)

    return (1.0/(len(t1)**2))*sum1-(1.0/(len(t0)**2))*sum0



if __name__=='__main__':  #只有在执行当前模块时才会运行此函数

    # 年龄匹配数据集的核方法
    agesonly = linearclassify.loadmatch('agesonly.csv',allnum=True)    #读入只关注年龄的配对情况
    print(agesonly)
    offset = getoffset(agesonly)   #获取高维度下的数据偏移
    print(offset)
    result = nlclassify([30,30],agesonly,offset)   #使用核方法来判断属于哪个分类
    print(result)

    # 复杂数据集的核方法
    numercalset = linearclassify.loadnumerical()  # 获取转化为数值型的复杂数据集
    scaledset, scalef = linearclassify.scaledata(numercalset)  # 对复杂数据集进行比例缩放
    ssoffset = getoffset(scaledset)   #获取高维度下的数据偏移
    onedata = scalef([28.0,-1,-1,26.0,-1,1,2,0.8])  # 取一个数据作为新数据先比例缩放
    result = nlclassify(onedata, scaledset, ssoffset)  # 使用核方法来判断属于哪个分类
    print(result)