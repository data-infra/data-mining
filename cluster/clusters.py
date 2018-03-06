# -- coding: utf-8 --
#读取数据源，对数据聚类
from PIL import Image,ImageDraw

# 对输入对象（行）进行分类，并不知道具体是什么类，只知道哪些对象属于一组。所以是非监督型的
# 每行代表一个输入对象，每列代表对象的一个特征属性。取值用数字型，以便计算距离。


# 读取表格型数据，获取特征数据集。
def readfile(filename):
    lines=[line for line in open(filename)]

    # 第一行是列标题
    colnames=lines[0].strip().split('\t')[1:]
    rownames=[]
    data=[]
    for line in lines[1:]:
        p=line.strip().split('\t')
        # 每行的第一列是行名
        rownames.append(p[0])
        # 剩余部分就是该行对应的数据
        onerow = [float(x) for x in p[1:]]
        data.append(onerow)
    return rownames,colnames,data


from math import sqrt

# 计算两个聚类中心点的皮尔逊相似度
def pearson(v1,v2):
    # 简单求和
    sum1=sum(v1)
    sum2=sum(v2)

    # 求平方和
    sum1Sq=sum([pow(v,2) for v in v1])
    sum2Sq=sum([pow(v,2) for v in v2])

    # 求乘积之和
    pSum=sum([v1[i]*v2[i] for i in range(len(v1))])

    # 计算r
    num=pSum-(sum1*sum2/len(v1))
    den=sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
    if den==0: return 0

    return 1.0-num/den


# 定义一个聚类，包含左右子聚类。
class bicluster:
    def __init__(self,vec,left=None,right=None,distance=0.0,id=None):
        self.left=left    #左子聚类
        self.right=right  #右子聚类
        self.vec=vec      #聚类的中心点
        self.id=id        #聚类的id
        self.distance=distance  #左右子聚类间的距离（相似度）


# 根据数据集形成聚类树
def hcluster(rows,distance=pearson):
    distance_set={}
    currentclustid=-1

    # 最开始聚类就是数据集中的行，每行一个聚类
    clust=[bicluster(rows[i],id=i) for i in range(len(rows))]   #原始集合中的聚类都设置了不同的正数id，（使用正数是为了标记这是一个叶节点）（使用不同的数是为了建立配对集合）

    while len(clust)>1:
        lowestpair=(0,1)       # lowestpair用来存储距离最小的一对聚类的索引对
        closest=distance(clust[0].vec,clust[1].vec)  # closest用来存储最小距离

        # 遍历每一对聚类，寻找距离最小的一对聚类
        for i in range(len(clust)):
            for j in range(i+1,len(clust)):
                # 用distance_set来缓存距离最小的计算值
                if (clust[i].id,clust[j].id) not in distance_set:
                    distance_set[(clust[i].id,clust[j].id)]=distance(clust[i].vec,clust[j].vec)

                d=distance_set[(clust[i].id,clust[j].id)]

                if d<closest:
                    closest=d
                    lowestpair=(i,j)

        # 计算距离最近的两个聚类的平均值作为代表新聚类的中心点
        mergevec=[(clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0 for i in range(len(clust[0].vec))]

        # 将距离最近的两个聚类合并成新的聚类
        newcluster=bicluster(mergevec,left=clust[lowestpair[0]],
                             right=clust[lowestpair[1]],
                             distance=closest,id=currentclustid)

        # 不再原始集合中的聚类id设置为负数。为了标记这是一个枝节点
        currentclustid-=1
        # 删除旧的聚类。（因为旧聚类已经添加为新聚类的左右子聚类了）
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)

    return clust[0]   #返回聚类树


# 将聚类树以类似文件系统层级结构的形式打印输出
def printclust(clust,labels=None,n=0):
    # 利用缩进来建立层级布局
    for i in range(n): print(' ',end='')
    if clust.id<0:
        # 负数标记代表这是一个分支
        print('-')
    else:
        # 正数标记代表这是一个叶节点
        if labels==None: print(clust.id)
        else: print(labels[clust.id])

    # 现在开始打印右侧分支和左侧分支
    if clust.left!=None: printclust(clust.left,labels=labels,n=n+1)
    if clust.right!=None: printclust(clust.right,labels=labels,n=n+1)


# 绘制树状图——获取聚类树要显示完整需要的高度
def getheight(clusttree):
    # 若是叶节点则高度为1
    if clusttree.left==None and clusttree.right==None: return 1

    # 否则，高度为左右分枝的高度之和
    return getheight(clusttree.left)+getheight(clusttree.right)

# 绘制树状图——获取聚类树要显示完整需要的深度（宽度）
def getdepth(clusttree):
    # 一个叶节点的距离是0.0
    if clusttree.left==None and clusttree.right==None: return 0

    # 一个叶节点的距离=左右两侧分支中距离较大者 + 该支节点自身的距离
    return max(getdepth(clusttree.left),getdepth(clusttree.right))+clusttree.distance


# 画聚类节点以及子聚类节点
def drawnode(draw,clust,x,y,scaling,labels):
    if clust.id<0:
        h1=getheight(clust.left)*20
        h2=getheight(clust.right)*20
        top=y-(h1+h2)/2
        bottom=y+(h1+h2)/2
        # 线的长度
        ll=clust.distance*scaling
        # 聚类到其子节点的垂直线
        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))

        # 连接左侧节点的水平线
        draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))

        # 连接右侧节点的水平线
        draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))

        # 调用函数绘制左右子节点
        drawnode(draw,clust.left,x+ll,top+h1/2,scaling,labels)
        drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,labels)
    else:
        # 如果这是一个叶节点，则绘制节点的标签文本
        draw.text((x+5,y-7),labels[clust.id],(0,0,0))

# 绘制树状图——为每一个最终生成的聚类创建一个高度为20像素，宽度固定的图片。其中缩放因子是由固定宽度除以总的深度得到的
def drawdendrogram(clusttree, labels, jpeg='clusters.jpg'):
    # 高度和宽度
    h = getheight(clusttree) * 20
    w = 1200
    depth = getdepth(clusttree)

    # 由于宽度是固定的，因此我们需要对距离值做相应的调整。（因为显示窗口宽度固定，高度可上下拖动）
    scaling = float(w - 150) / depth

    # 新建一个白色背景的图片
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.line((0, h / 2, 10, h / 2), fill=(255, 0, 0))

    # 画根节点（会迭代调用画子节点）
    drawnode(draw, clusttree, 10, (h / 2), scaling, labels)
    img.save(jpeg, 'JPEG')



# 数据集转置，进行列聚类。
def rotatematrix(data):
    newdata=[]
    for i in range(len(data[0])):
        newrow=[data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata  #一行表示一个单词在每篇博客中出现的次数






import random
# k均值聚类：随机设置k个聚类点，根据每个点到k个聚类点的距离，为每个点分组。移动聚类点到组成员中心位置。重新计算重新分组重新移动。迭代至成员不再变化。
# 每个点代表一行，每个聚类点，代表一类。参数：数据集，距离计算算法，聚类的数目。
def kcluster(rows,distance=pearson,k=4):
    # 确定每个点的特征的最小值和最大值
    ranges=[(min([row[i] for row in rows]),max([row[i] for row in rows]))  for i in range(len(rows[0]))]

    # 随机创建K个中心点
    clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0]
               for i in range(len(rows[0]))] for j in range(k)]

    lastmatches=None
    for t in range(100):   #默认迭代100次。
        print('迭代 %d' % t)
        bestmatches=[[] for i in range(k)]  #生成k个空数组，用于存储k个聚类点包含的成员

        # 在每一行中寻找距离最近的中心点
        for j in range(len(rows)):
            row=rows[j]
            bestmatch=0
            for i in range(k):
                d=distance(clusters[i],row)
                if d<distance(clusters[bestmatch],row): bestmatch=i   #计算与哪个聚类点最近
            bestmatches[bestmatch].append(j)  #每个聚类点记录它包含的组成员

        # 如果结果与上一次相同，则整个过程结束
        if bestmatches==lastmatches: break
        lastmatches=bestmatches

        # 把中心点移动到成员的平均位置处
        for i in range(k):
            avgs=[0.0]*len(rows[0])
            if len(bestmatches[i])>0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m]+=rows[rowid][m]
                for j in range(len(avgs)):   #在每个维度都计算均值
                    avgs[j]/=len(bestmatches[i])
                clusters[i]=avgs

    return bestmatches   #返回k个聚类点，以及所包含的所有成员

# 使用tanimoto系数表示距离（相似度）：两个向量的交集/两个向量的并集
def tanamoto(v1,v2):
    c1,c2,shr=0,0,0
    for i in range(len(v1)):
        if v1[i]!=0: c1+=1 # 出现在v1中
        if v2[i]!=0: c2+=1 # 出现在v2中
        if v1[i]!=0 and v2[i]!=0: shr+=1 # 在两个向量中都出现

    return 1.0-(float(shr)/(c1+c2-shr))

# 多维缩放技术：将数据集（多维）以二维的形式表达（原理：在二维图中的距离等比与真实的多维距离即可）。
def scaledown(data,distance=pearson,rate=0.01):
    n=len(data)

    # 每一对数据项之间的真实距离
    realdist=[[distance(data[i],data[j]) for j in range(n)] for i in range(0,n)]

    # 随机初始化节点在二维空间中的起始位置
    loc=[[random.random(),random.random()] for i in range(n)]
    fakedist=[[0.0 for j in range(n)] for i in range(n)]

    lasterror=None
    for m in range(0,1000):
        # 计算在二维图中的距离
        for i in range(n):
            for j in range(n):
                fakedist[i][j]=sqrt(sum([pow(loc[i][x]-loc[j][x],2) for x in range(len(loc[i]))]))

        # 根据真实多维距离和二维距离间的误差，移动节点
        grad=[[0.0,0.0] for i in range(n)]

        totalerror=0
        for k in range(n):
            for j in range(n):
                if j==k: continue
                # 误差值等于目标距离与当前距离之间的差值的百分比
                errorterm=(fakedist[j][k]-realdist[j][k])/realdist[j][k]

                # 每一个节点都需要根据误差的多少，按比例远离或靠近其他节点
                grad[k][0]+=((loc[k][0]-loc[j][0])/fakedist[j][k])*errorterm
                grad[k][1]+=((loc[k][1]-loc[j][1])/fakedist[j][k])*errorterm

                # 记录总的误差值
                totalerror+=abs(errorterm)
        # print(totalerror)

        # 如果节点移动之后的情况变得更糟，则程序结束
        if lasterror and lasterror<totalerror: break
        lasterror=totalerror

        # 根据rate参数与grad值相乘的结果，移动每一个节点
        for k in range(n):
            loc[k][0]-=rate*grad[k][0]
            loc[k][1]-=rate*grad[k][1]

    return loc


# 生成二维位置图
def draw2d(data,labels,jpeg='mds2d.jpg'):
    img=Image.new('RGB',(2000,2000),(255,255,255))
    draw=ImageDraw.Draw(img)
    for i in range(len(data)):
        x=(data[i][0]+0.5)*1000
        y=(data[i][1]+0.5)*1000
        draw.text((x,y),labels[i],(0,0,0))
    img.save(jpeg,'JPEG')



if __name__=='__main__':
    blognames,words,data = readfile('blogdata.txt')  #加载数据集
    # clust = hcluster(data)  #构建层次聚类树
    # # printclust(clust,labels=blognames)  # 打印聚类树
    # drawdendrogram(clust,blognames,jpeg='blogclust.jpg')  # 绘制层次聚类树
    # #
    # #
    # rdata = rotatematrix(data)  #旋转数据集矩阵，对特征属性进行聚类
    # wordclust = hcluster(rdata) #构建特征的聚类树
    # drawdendrogram(wordclust,labels=words,jpeg='wordclust.jpg')  # 绘制特征的层次聚类树


    kclust = kcluster(data,k=10)  #k均值聚类，形成k个聚类
    print([blognames[r] for r in kclust[0]])  # 打印第一个聚类的所有博客名称


    # 多维数据的二维可视化
    # coords = scaledown(data)
    # draw2d(coords,blognames,jpeg='blogs2d.jpg')

