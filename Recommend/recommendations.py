#根据偏好数据集，匹配行和列，进而实现推荐行和推荐列

# 偏好数据集（人-电影-评分）
prefs={
    'name1': {'movie1': 2.5, 'movie2': 3.5,'movie3': 3.0, 'movie4': 3.5, 'movie5': 2.5, 'movie6': 3.0},
    'name2': {'movie1': 3.0, 'movie2': 3.5,'movie3': 1.5, 'movie4': 5.0, 'movie6': 3.0,'movie5': 3.5},
    'name3': {'movie1': 2.5, 'movie2': 3.0,'movie4': 3.5, 'movie6': 4.0},
    'name4': {'movie2': 3.5, 'movie3': 3.0, 'movie6': 4.5, 'movie4': 4.0, 'movie5': 2.5},
    'name5': {'movie1': 3.0, 'movie2': 4.0, 'movie3': 2.0, 'movie4': 3.0, 'movie6': 3.0,'movie5': 2.0},
    'name6': {'movie1': 3.0, 'movie2': 4.0, 'movie6': 3.0, 'movie4': 5.0, 'movie5': 3.5},
    'name7': {'movie2':4.5,'movie5':1.0,'movie4':4.0}
}


from math import sqrt
# 计算两行之间的欧几里得距离，以此来代表相似度。prefs表示偏好数据集
def sim_distance(prefs,row1_name,row2_name):
    # 首先计算是否有共同列（都看过的电影）
    si={}
    for item in prefs[row1_name]:
        if item in prefs[row2_name]: si[item]=1

    # 如果没有共同列，则两行之间相似度为0
    if len(si)==0: return 0

    # 根据共同列计算两行的欧几里得距离，并将距离映射到0-1上。0表示完全不相似，1表示完全相似
    sum_of_squares=sum([pow(prefs[row1_name][item]-prefs[row2_name][item],2) for item in prefs[row1_name] if item in prefs[row2_name]])
    return 1/(1+sum_of_squares)

# 计算两行的皮尔逊相似度，以此来代表相似度。prefs表示数据集
def sim_pearson(prefs,row1_name,row2_name):
    # 首先计算是否有共同列（都看过的电影）
    si={}
    for item in prefs[row1_name]:
        if item in prefs[row2_name]: si[item]=1

        # 如果没有共同列，两行之间相似度为0
    if len(si)==0: return 0

    # 得到列表元素个数
    n=len(si)

    # 对两行的共同列求和
    sum1=sum([prefs[row1_name][it] for it in si])
    sum2=sum([prefs[row2_name][it] for it in si])

    # 对两行的共同列求平方和
    sum1Sq=sum([pow(prefs[row1_name][it],2) for it in si])
    sum2Sq=sum([pow(prefs[row2_name][it],2) for it in si])

    # 对两行的共同列求乘积之和
    pSum=sum([prefs[row1_name][it]*prefs[row2_name][it] for it in si])

    # 计算皮尔逊评价值
    num=pSum-(sum1*sum2/n)
    den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den==0: return 0

    r=num/den

    return r

# 匹配相似行
# 根据偏好数据集，返回与某个行最匹配的n行。person表示要匹配的行（人），similarity表示相似度计算函数
def topMatches(prefs,row_name,n=5,similarity=sim_pearson):
    scores=[(similarity(prefs,row_name,other),other) for other in prefs if other!=row_name]
    scores.sort()
    scores.reverse()
    num = n
    if n>len(scores):num= len(scores)
    return scores[0:num]

# 利用相似行，估计某行所有列存在的空白值，并排名（估计影片评分，并排名推荐）
# 利用所有其他行的各列取值的加权平均（相似度为权值），为某行各列提供估值
def getRecommendations(prefs,row_name,similarity=sim_pearson):
    totals={}
    simSums={}
    for other in prefs:
        # 不和自己做比较
        if other==row_name: continue
        sim=similarity(prefs,row_name,other)

        # 忽略评价值为0或为负的情况
        if sim<=0: continue
        for item in prefs[other]:
            # 只对自己还未有的列进行临时估值
            if item not in prefs[row_name] or prefs[row_name][item]==0:
                # 相似度*临时估值
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # 相似度之和
                simSums.setdefault(item,0)
                simSums[item]+=sim

    # 建立归一化列表
    rankings=[(total/simSums[item],item) for item,total in totals.items()]

    # 返回最终估值经过排序的列表
    rankings.sort()
    rankings.reverse()
    return rankings

# 转置偏好数据集，以便实现匹配列
def transformPrefs(prefs):
    result={}
    for row_name in prefs:
        for item in prefs[row_name]:
            result.setdefault(item,{})

            # 将行与列对调
            result[item][row_name]=prefs[row_name][item]
    return result

# 匹配相似列，返回各列的匹配集合（因为各列的匹配可提前在用户登陆前完成），
# 根据转置后的偏好数据集，获取每列相似的n个其他列
def calculateSimilarItems(prefs,n=10):
    # 建立字典，以给出与这些列最为相近的所有其他列
    itemMatch={}
    # 以列为中心对偏好矩阵实施转置处理
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
        # 针对大数据集更新状态变量
        c+=1
        if c%100==0: print("%d / %d" % (c,len(itemPrefs)))
        # 寻找最为相近的列
        scores=topMatches(itemPrefs,item,n=n,similarity=sim_distance)
        itemMatch[item]=scores

    return itemMatch    #返回每列匹配的其他列

# 利用相似列，对某一行的各列进行估值，（估计影片评分，并排名推荐）：根据偏好数据集和提前构造好的物品匹配库，向用户推荐物品
def getRecommendedItems(prefs,itemMatch,row_name):
    onerow=prefs[row_name]  #获取当前行所拥有的列
    scores={}
    totalSim={}
    # 循环遍历由当前行所拥有的列
    for (item,rating) in onerow.items( ):

        # 循环遍历与当前列相似的列
        for (similarity,item2) in itemMatch[item]:

            # 忽略行已经拥有的列
            if item2 in onerow: continue
            # 估值与相似度的加权之和
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # 全部相似度之和
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity

    # 将每个合计值除以加权和，求出平均值
    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]

    # 按最高值到最低值的顺序，返回估值排行
    rankings.sort( )
    rankings.reverse( )
    return rankings

#下载大量数据集
def loadMovieLens(path='/data/movielens'):
    # 获取影片标题
    movies={}
    for line in open(path+'/u.item'):
        (id,title)=line.split('|')[0:2]
        movies[id]=title

    # 加载数据
    prefs={}
    for line in open(path+'/u.data'):
        (user,movieid,rating,ts)=line.split('\t')
        prefs.setdefault(user,{})
        prefs[user][movies[movieid]]=float(rating)
    return prefs


if __name__=="__main__":     #只有在执行当前模块时才会运行此函数
    #利用相似人推荐相似物品
    rankings = getRecommendations(prefs,'name7')
    print(rankings)   #打印推荐排名
    #利用相似物品推荐相似物品
    itemMatch = calculateSimilarItems(prefs)  # 提前计算所有物品的相似物品
    rankings = getRecommendedItems(prefs,itemMatch,'name7')
    print(rankings)  #打印推荐排名