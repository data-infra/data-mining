from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# =======================================================频繁项集===================================================

# 获取候选1项集，dataSet为事务集。返回一个list，每个元素都是set集合
def createC1(dataSet):
    C1 = []   # 元素个数为1的项集（非频繁项集，因为还没有同最小支持度比较）
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()  # 这里排序是为了，生成新的候选集时可以直接认为两个n项候选集前面的部分相同
    # 因为除了候选1项集外其他的候选n项集都是以二维列表的形式存在，所以要将候选1项集的每一个元素都转化为一个单独的集合。
    return list(map(frozenset, C1))   #map(frozenset, C1)的语义是将C1由Python列表转换为不变集合（frozenset，Python中的数据结构）




# 找出候选集中的频繁项集
# dataSet为全部数据集，Ck为大小为k（包含k个元素）的候选项集，minSupport为设定的最小支持度
def scanD(dataSet, Ck, minSupport):
    ssCnt = {}   # 记录每个候选项的个数
    for tid in dataSet:
        for can in Ck:
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can, 0) + 1   # 计算每一个项集出现的频率
    numItems = float(len(dataSet))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)  #将频繁项集插入返回列表的首部
        supportData[key] = support
    return retList, supportData   #retList为在Ck中找出的频繁项集（支持度大于minSupport的），supportData记录各频繁项集的支持度


# 通过频繁项集列表Lk和项集个数k生成候选项集C(k+1)。
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 前k-1项相同时，才将两个集合合并，合并后才能生成k+1项
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]   # 取出两个集合的前k-1个元素
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

# 获取事务集中的所有的频繁项集
# Ck表示项数为k的候选项集，最初的C1通过createC1()函数生成。Lk表示项数为k的频繁项集，supK为其支持度，Lk和supK由scanD()函数通过Ck计算而来。
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)  # 从事务集中获取候选1项集
    D = list(map(set, dataSet))  # 将事务集的每个元素转化为集合
    L1, supportData = scanD(D, C1, minSupport)  # 获取频繁1项集和对应的支持度
    L = [L1]  # L用来存储所有的频繁项集
    k = 2
    while (len(L[k-2]) > 0): # 一直迭代到项集数目过大而在事务集中不存在这种n项集
        Ck = aprioriGen(L[k-2], k)   # 根据频繁项集生成新的候选项集。Ck表示项数为k的候选项集
        Lk, supK = scanD(D, Ck, minSupport)  # Lk表示项数为k的频繁项集，supK为其支持度
        L.append(Lk);supportData.update(supK)  # 添加新频繁项集和他们的支持度
        k += 1
    return L, supportData



# =====================================================关联规则==================================================
# 对候选规则集进行评估
# 计算规则的可信度，并过滤出满足最小可信度要求的规则。freqSet为频繁2项集， H为频繁项集每个元素都是集合的形式
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # 支持度相减就等级差集的置信度
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))  # 记录关联规则和置信度
            prunedH.append(conseq)  #记录关联规则的源项
    return prunedH

# 生成候选规则集。freqSet为频繁n项集，H为频繁项集每个元素多是集合的形式，brl为关联规则，minConf为最低置信区间
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])  #先计算要计算规则的源项的大小
    if (len(freqSet) > (m + 1)):  #查看该频繁项集是否大到可以移除大小为m的子集，也就是说是否还有目的项
        Hmpl = aprioriGen(H, m + 1)  # 生成候选集
        Hmpl = calcConf(freqSet, Hmpl, supportData, brl, minConf)  # 计算关联规则
        if (len(Hmpl) > 1):
            rulesFromConseq(freqSet, Hmpl, supportData, brl, minConf)  # 下一次迭代的列表


# 获取关联规则。L为频繁项集（不包含频繁1项集），supportData为频繁项集对象的支持度，minConf最小可信度阈值
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):  # 遍历每一个频繁项
        for freqSet in L[i]:  # 遍历每一个项
            H1 = [frozenset([item]) for item in freqSet]  #将每一个项转化为集合
            if (i > 1):
                # 三个及以上元素的集合
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                # 两个元素的集合
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


if __name__=='__main__':
    dataSet = loadDataSet()  # 获取事务集。每个元素都是列表
    # C1 = createC1(dataSet)  # 获取候选1项集。每个元素都是集合
    # D = list(map(set, dataSet))  # 转化事务集的形式，每个元素都转化为集合。
    # L1, suppDat = scanD(D, C1, 0.5)
    # print(L1,suppDat)


    # L, suppData = apriori(dataSet,minSupport=0.7)
    # print(L,suppData)


    L, suppData = apriori(dataSet, minSupport=0.5)
    print(L,suppData)
    rules = generateRules(L, suppData, minConf=0.7)
    print(rules)