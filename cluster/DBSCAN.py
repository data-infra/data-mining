import pylab as pl
from collections import defaultdict,Counter

# 加载数据集
def loaddata(filepath):
    points = [[float(eachpoint.split(",")[0]), float(eachpoint.split(",")[1])] for eachpoint in open(filepath,"r")]
    return points

# 以距离最大的维度上的距离为两个对象之间的距离
def distance(point1,point2):
    return max(abs(point1[0] - point2[0]),abs(point1[1] - point2[1]))


# 计算每个数据点相邻的数据点，邻域定义为以该点为中心以边长为2*EPs的网格
def getSurroundPoint(points,Eps=1):
    surroundPoints = {}  # 每个元素默认是一个空列表
    for idx1,point1 in enumerate(points):
        for idx2,point2 in enumerate(points):
            if (idx1 < idx2):
                if(distance(point1,point2)<=Eps):
                    surroundPoints.setdefault(idx1,[])   # 设置每个点的默认值邻节点为空列表
                    surroundPoints.setdefault(idx2, [])   # 设置每个点的默认值邻节点为空列表
                    surroundPoints[idx1].append(idx2)
                    surroundPoints[idx2].append(idx1)
    return surroundPoints



# 定义邻域内相邻的数据点的个数大于4的为核心点，获取核心点以及核心点的周边点
def findallCore(points,surroundPoints,Eps=10,MinPts=5):
    # 获取所有核心点
    corePointIdx = [pointIdx for pointIdx,surPointIdxs in surroundPoints.items() if len(surPointIdxs)>=MinPts]
    # 邻域内包含某个核心点的非核心点，定义为边界点
    borderPointIdx = []
    for pointIdx,surPointIdxs in surroundPoints.items():
        if (pointIdx not in corePointIdx):  # 边界点本身不是核心点
            for onesurPointIdx in surPointIdxs:
                if onesurPointIdx in corePointIdx:  # 边界点周边至少包含一个核心点
                    borderPointIdx.append(pointIdx)
                    break

    corePoint = [points[pointIdx] for pointIdx in corePointIdx]  # 核心点
    borderPoint = [points[pointIdx] for pointIdx in borderPointIdx]  #边界点
    return corePointIdx,borderPointIdx

# 获取所有噪声点。噪音点既不是边界点也不是核心点
def findallnoise(points,corePointIdx,borderPointIdx):
    noisePointIdx = [pointIdx for pointIdx in range(len(points)) if pointIdx not in corePointIdx and pointIdx not in borderPointIdx]
    noisePoint = [points[pointIdx] for pointIdx in noisePointIdx]
    return noisePoint




# 根据邻域关系，核心点，边界点进行分簇
def divideGroups(points,surroundPoints,corePointIdx,borderPointIdx):
    groups = [idx for idx in range(len(points))]  # groups记录每个节点所属的簇编号
    # 各个核心点与其邻域内的所有核心点放在同一个簇中
    for pointidx,surroundIdxs in surroundPoints.items():
        for oneSurroundIdx in surroundIdxs:
            if (pointidx in corePointIdx and oneSurroundIdx in corePointIdx and pointidx < oneSurroundIdx):
                for idx in range(len(groups)):
                    if groups[idx] == groups[oneSurroundIdx]:
                        groups[idx] = groups[pointidx]

    # 边界点跟其邻域内的某个核心点放在同一个簇中
    for pointidx,surroundIdxs in surroundPoints.items():
        for oneSurroundIdx in surroundIdxs:
            if (pointidx in borderPointIdx and oneSurroundIdx in corePointIdx):
                groups[pointidx] = groups[oneSurroundIdx]
                break
    return groups

# 绘制分簇图
def plotgroup(points,groups,noisePoint):
    # 取簇规模最大的3个簇
    finalGroup = Counter(groups).most_common(3)
    finalGroup = [onecount[0] for onecount in finalGroup]
    group1 = [points[idx] for idx in range(len(points)) if groups[idx]==finalGroup[0]]
    group2 = [points[idx] for idx in range(len(points)) if groups[idx]==finalGroup[1]]
    group3 = [points[idx] for idx in range(len(points)) if groups[idx]==finalGroup[2]]
    pl.plot([eachpoint[0] for eachpoint in group1], [eachpoint[1] for eachpoint in group1], 'or')
    pl.plot([eachpoint[0] for eachpoint in group2], [eachpoint[1] for eachpoint in group2], 'oy')
    pl.plot([eachpoint[0] for eachpoint in group3], [eachpoint[1] for eachpoint in group3], 'og')
    # 打印噪音点，黑色
    pl.plot([eachpoint[0] for eachpoint in noisePoint], [eachpoint[1] for eachpoint in noisePoint], 'ok')
    pl.show()


if __name__=='__main__':
    points = loaddata('DBSCAN_data.txt')   # 加载数据
    surroundPoints=getSurroundPoint(points,Eps=2)  # 获取邻域关系
    corePointIdx, borderPointIdx = findallCore(points,surroundPoints,Eps=2,MinPts=3)  # 获取核心节点和边界节点
    noisePoint = findallnoise(points,corePointIdx,borderPointIdx)  # 获取噪音节点
    groups = divideGroups(points,surroundPoints,corePointIdx,borderPointIdx)   # 节点分簇
    plotgroup(points, groups, noisePoint)  # 可视化绘图