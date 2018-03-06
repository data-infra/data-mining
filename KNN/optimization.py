# 优化算法。寻找使成本函数最小的题解。精髓：1、将题解转化为数字序列化，可以写出题解范围。2、成本函数能返回值
import time
import random
import math


# 随机搜索算法：随机选择题解，计算成本值，成本值最小的题解为确定题解。domain为题解范围（可选航班范围），costf为成本函数。
def randomoptimize(domain,costf):
    best=999999999
    bestr=None
    for i in range(0,1000):
        # 创建随机解
        sol=[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        #计算成本值
        cost=costf(sol)

        # 与目前得到的最优解进行比较
        if cost<best:
            best=cost
            bestr=sol
    return sol  #返回随机最优解


# 爬山法：对于成本函数连续的情况，题解向成本值减小的地方渐变，直到成本值不再变化。domain为题解范围（可选航班范围），costf为成本函数。
# 在只有一个极低点时最有效。可以采用随机重复爬山法优化
def hillclimb(domain,costf):
    # 创建一个随机解
    sol=[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
    # 主循环
    while 1:
        # 创建相邻解的列表
        neighbors=[]

        for j in range(len(domain)):   #在j等于0和末尾的时候存在问题
            # 在每个方向上相对于原值偏离一点。每个方向上的偏离不相互影响
            if sol[j]>domain[j][0]:
                neighbors.append(sol[0:j]+[sol[j]-1]+sol[j+1:])  #向近0偏移
            if sol[j]<domain[j][1]:
                neighbors.append(sol[0:j]+[sol[j]+1]+sol[j+1:])  #远0偏移

        # 在相邻解中寻找最优解
        current=costf(sol)
        best=current
        for j in range(len(neighbors)):
            cost=costf(neighbors[j])
            if cost<best:
                best=cost
                sol=neighbors[j]

        # 如果没有更好的解，则退出循环。即寻找了极低点
        if best==current:
            break
    return sol

# 模拟退火算法：概率性接收更优解（更差解），时间越久，概率越大（越低）。
def annealingoptimize(domain,costf,T=10000.0,cool=0.95,step=1):
    # 随机初始化值
    vec=[float(random.randint(domain[i][0],domain[i][1])) for i in range(len(domain))]

    while T>0.1:
        # 选择一个索引值
        i=random.randint(0,len(domain)-1)

        # 选择一个改变索引值的方向
        dir=random.randint(-step,step)

        #创建一个代表题解的新列表，改变其中一个值
        vecb=vec[:]
        vecb[i]+=dir
        if vecb[i]<domain[i][0]: vecb[i]=domain[i][0]   #如果渐变不超出了题解的范围
        elif vecb[i]>domain[i][1]: vecb[i]=domain[i][1] #如果渐变不超出了题解的范围

        # 计算当前成本与新的成本
        ea=costf(vec)
        eb=costf(vecb)
        p=pow(math.e,(-eb-ea)/T)

        # 它是更好的解么？或者是趋向最优解的可能的临界解么
        if (eb<ea or random.random()<p):
            vec=vecb

            # 降低温度
        T=T*cool
    return vec

# 遗传算法：基因杂交（交叉）或基因变异。domain题解范围，costf成本函数，popsize种群大小，step变异基因量，mutprob变异比例，elite优秀基因者的比例，maxiter运行多少代
def geneticoptimize(domain,costf,popsize=50,step=1,mutprob=0.2,elite=0.2,maxiter=100):
    # 变异操作，存在变异失败的情况
    def mutate(vec):
        i=random.randint(0,len(domain)-1)
        if random.random()<0.5 and vec[i]>=domain[i][0]+step:
            return vec[0:i]+[vec[i]-step]+vec[i+1:]
        elif vec[i]<=domain[i][1]-step:
            return vec[0:i]+[vec[i]+step]+vec[i+1:]

    # 杂交操作（交叉）
    def crossover(r1,r2):
        i=random.randint(1,len(domain)-2)
        return r1[0:i]+r2[i:]

    # 构建初始种群
    pop=[]
    for i in range(popsize):   #随机产生50个动物的种群
        vec=[random.randint(domain[i][0],domain[i][1]) for i in range(len(domain))]
        pop.append(vec)
    # 每一代有多少胜出者？
    topelite=int(elite*popsize)

    # 主循环
    for i in range(maxiter):
        scores=[(costf(v),v) for v in pop]
        scores.sort()
        ranked=[v for (s,v) in scores]

        # 在种群中选出优胜者
        pop=ranked[0:topelite]

        # 为优秀基因者，添加变异和配对后的胜出者
        while len(pop)<popsize:
            if random.random()<mutprob:   #变异所占的比例

                # 变异
                c=random.randint(0,topelite-1)
                newanimal = mutate(ranked[c])
                if newanimal!=None:    #有可能存在变异死亡者
                    pop.append(newanimal)
            else:

                # 相互杂交。以后会遇到近亲结婚的问题
                c1=random.randint(0,topelite-1)
                c2=random.randint(0,topelite-1)
                newanimal = crossover(ranked[c1], ranked[c2])
                pop.append(newanimal)


        # 打印当前最优解和成本
        # print(scores[0])
    return scores[0][1]  #返回最优解



