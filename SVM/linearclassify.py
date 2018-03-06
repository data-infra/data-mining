# 线性分类器：计算样本数据每个分类中所有节点的平均值。对新输入对象计算到哪个中心点最近就属于哪个分类
# 使用基本线性分类进行婚姻数据匹配

# 场景：男女不同的属性信息，例如年龄、是否吸烟、是否要孩子、兴趣列表、家庭住址。产生的输出结果，配对成功还是不成功

# 定义数据类
class matchrow:
    def __init__(self,row,allnum=False):
        if allnum:
            self.data=[float(row[i]) for i in range(len(row)-1)]   #如果每个属性都是数字就转化为浮点型
        else:
            self.data=row[0:len(row)-1]  #如果并不是数字，就保留源数据类型
        self.matchresult=int(row[len(row)-1])   #最后一位表示分类（匹配结果），0表示匹配失败，1表示匹配成功

# 从文件中加载数据.allnum表示是否所有属性都是数字
def loadmatch(filename,allnum=False):
    rows=[]
    for line in open(filename):
        rows.append(matchrow(line.split(','),allnum))
    return rows



import matplotlib.pyplot as plt

# 绘制只根据年龄进行配对的结果分布散点图
def plotagematches(rows):
    xdm,ydm=[r.data[0] for r in rows if r.matchresult==1],[r.data[1] for r in rows if r.matchresult==1]
    xdn,ydn=[r.data[0] for r in rows if r.matchresult==0],[r.data[1] for r in rows if r.matchresult==0]

    plt.plot(xdm,ydm,'bo')
    plt.plot(xdn,ydn,'b+')

    plt.show()


# 使用基本的线性分类。rows为样本数据集。（计算样本数据每个分类中所有节点的平均值。对新输入对象计算到哪个中心点最近就属于哪个分类）
def lineartrain(rows):
    averages={}
    counts={}

    for row in rows:
        # 得到该坐标点所属的分类
        cat=row.matchresult

        averages.setdefault(cat,[0.0]*(len(row.data)))
        counts.setdefault(cat,0)

        # 将该坐标点加入averages中。每个维度都要求均值
        for i in range(len(row.data)):
            averages[cat][i]+=float(row.data[i])

        # 记录每个分类中有多少个坐标点
        counts[cat]+=1

    # 将总和除以计数值以求得平均值
    for cat,avg in averages.items():
        for i in range(len(avg)):
            avg[i]/=counts[cat]

    return averages

# 绘制线性分类器均值点和分割线
def plotlinear(rows):
    xdm,ydm=[r.data[0] for r in rows if r.matchresult==1],[r.data[1] for r in rows if r.matchresult==1]
    xdn,ydn=[r.data[0] for r in rows if r.matchresult==0],[r.data[1] for r in rows if r.matchresult==0]

    plt.plot(xdm,ydm,'bo')
    plt.plot(xdn,ydn,'b+')
    # 获取均值点
    averages = lineartrain(rows)
    #绘制均值点
    averx=[]
    avery=[]
    for value in averages.values():
        averx.append(value[0])
        avery.append(value[1])

        plt.plot(averx,avery,'r*')
    #绘制垂直平分线作为分割线
    # y=-(x1-x0)/(y1-y0)* (x-(x0+x1)/2)+(y0+y1)/2
    xnew = range(15,60,1)
    print(xnew)
    print(averx,avery)
    ynew = [-(averx[1]-averx[0])/(avery[1]-avery[0])*(x-(averx[0]+averx[1])/2)+(avery[0]+avery[1])/2 for x in xnew]
    plt.plot(xnew, ynew, 'r--')
    plt.axis([15, 52, 15, 50])  #设置显示范围
    plt.show()


# ================使用点积函数来代替欧几里德距离=================

# 向量点积函数，代替欧几里得距离
def dotproduct(v1,v2):
    return sum([v1[i]*v2[i] for i in range(len(v1))])

# 向量线段长度。
def veclength(v):
    return sum([p**2 for p in v])

# 使用点积结果为正还是负来判断属于哪个分类
def dpclassify(point,avgs):
    b=(dotproduct(avgs[1],avgs[1])-dotproduct(avgs[0],avgs[0]))/2
    y=dotproduct(point,avgs[0])-dotproduct(point,avgs[1])+b
    if y>0: return 0
    else: return 1




# ======================复杂数据集的线性分类器==========================

# 将是否问题转化为数值。yes转化为1，no转化为-1，缺失或模棱两可转化为0
def yesno(v):
    if v=='yes': return 1
    elif v=='no': return -1
    else: return 0

# 将列表转化为数值。获取公共项的数目。获取两个人相同的兴趣数量
def matchcount(interest1,interest2):
    l1=interest1.split(':')
    l2=interest2.split(':')
    x=0
    for v in l1:
        if v in l2: x+=1
    return x


# 利用百度地图来计算两个人的位置距离
baidukey="tc42noD8p3SO1hZhFTryMeRv"
import urllib
import json
# 使用geocoding api发起指定格式的请求，解析指定格式的返回数据，获取地址的经纬度
# http://api.map.baidu.com/geocoder/v2/?address=北京市海淀区上地十街10号&output=json&ak=您的ak&callback=showLocation
ak ='HIa8GVmtk9WSjhuevGfqMCGu'
loc_cache={}
def getlocation(address):   #这个结果每次获取最好存储在数据库中，不然每次运行都要花费大量的时间获取地址
    if address in loc_cache: return loc_cache[address]
    urlpath = 'http://api.map.baidu.com/geocoder/v2/?address=%s&output=json&ak=%s' % (urllib.parse.quote_plus(address),ak)
    data=urllib.request.urlopen(urlpath).read()
    response = json.loads(data,encoding='UTF-8')  # dict
    if not response['result']:
        print('没有找到地址：'+address)
        return None

    long = response['result']['location']['lng']
    lat = response['result']['location']['lat']
    loc_cache[address]=(float(lat),float(long))
    print('地址：' + address+"===经纬度："+str(loc_cache[address]))
    return loc_cache[address]

# 计算两个地点之间的实际距离
def milesdistance(a1,a2):
    try:
        lat1,long1=getlocation(a1)
        lat2,long2=getlocation(a2)
        latdif=69.1*(lat2-lat1)
        longdif=53.0*(long2-long1)
        return (latdif**2+longdif**2)**.5
    except:
        return None



# 构造新的数据集。包含各个复杂属性转化为数值数据
def loadnumerical():
    oldrows=loadmatch('matchmaker.csv')
    newrows=[]
    for row in oldrows:
        d=row.data
        distance = milesdistance(d[4],d[9])  # 以为有可能无法获取地址的经纬度，进而无法获取两地之间的距离，这里就成了缺失值。我们暂且直接抛弃缺失值
        if distance:
            data=[float(d[0]),yesno(d[1]),yesno(d[2]),
                  float(d[5]),yesno(d[6]),yesno(d[7]),
                  matchcount(d[3],d[8]),distance,row.matchresult]
            newrows.append(matchrow(data))
    return newrows


# 对数据进行缩放处理，全部归一化到0-1上，因为不同参考变量之间的数值尺度不同
def scaledata(rows):
    low=[999999999.0]*len(rows[0].data)
    high=[-999999999.0]*len(rows[0].data)
    # 寻找最大值和最小值
    for row in rows:
        d=row.data
        for i in range(len(d)):
            if d[i]<low[i]: low[i]=d[i]
            if d[i]>high[i]: high[i]=d[i]

    # 对数据进行缩放处理的函数
    def scaleinput(d):
        return [(d[i]-low[i])/(high[i]-low[i])
                for i in range(len(low))]

    # 对所有数据进行缩放处理
    newrows=[matchrow(scaleinput(row.data)+[row.matchresult]) for row in rows]

    # 返回新的数据和缩放处理函数
    return newrows,scaleinput






if __name__=='__main__':  #只有在执行当前模块时才会运行此函数

    agesonly = loadmatch('agesonly.csv')    #读入只关注年龄的配对情况
    # plotagematches(agesonly)    #绘制年龄配对散点图

    #======使用基本线性分类器分类===========
    # plotlinear(agesonly)   #绘制线性分类器均值点和分割线

    #==========复杂数据的线性分类器==========
    numercalset=loadnumerical()   #获取转化为数值型的复杂数据集
    scaledset,scalef=scaledata(numercalset)  #对复杂数据集进行比例缩放
    catavgs = lineartrain(scaledset)   #计算分类均值点
    print(catavgs)
    onedata = scalef(numercalset[0].data) #取一个数据作为新数据先比例缩放
    dpclassify(onedata,catavgs)   #使用点积结果来判断属于哪个分类
