# 优化算法。寻找使成本函数最小的题解。精髓：1、将题解转化为数字序列化，可以写出题解范围。2、成本函数能返回值
import time
import random
import math

# 问题场景：
# 所有乘客从不同的地方飞到同一个目的地，服务人员等待所有人到来以后将人一次性接走。
# 离开时，服务人员将人一次性带到飞机场，所有乘客等待自己的航班离开。
# 要解决的问题：
# 如何设置乘客的到来和离开航班，以及接送机的时间，使得总代价最小。

# 将题解设为数字序列。
# 数字表示某人乘坐的第几次航班，从0开始，例如[1,4,3,2,7,3,6,3,2]表示第1个人做第2个航班来，第5个航班走，第2个人做第4个航班来，第3个航班走

# 题解相互独立：组团旅游问题，举办会议的人员接送问题


# 人员的名称和来源地信息
peoplelist = [('name1','src1_place'),
              ('name2','src2_place'),
              ('name3','src3_place'),
              ('name4','src4_place'),
              ('name5','src5_place'),
              ('name6','src6_place')]
# 目的机场
destination='des_place'

flights={}  #加载所有航班信息。
# 查询所有的航班信息
for line in open('schedule.txt'):
    origin,dest,depart,arrive,price=line.strip().split(',')  #源地址、目的地址、离开时间、到达时间、价格
    flights.setdefault((origin,dest),[])   #航班信息已起止点为键值，每个起止点有多个航班

    # 将航班信息添加到航班列表中
    flights[(origin,dest)].append((depart,arrive,int(price)))  #按时间顺序扩展每次航班



# 将数字序列转换为航班，打印输出。输入为数字序列
def printschedule(r):
    for d in range(int(len(r)/2)):
        name=peoplelist[d][0]    #人员名称
        origin=peoplelist[d][1]  #人员来源地
        out=flights[(origin,destination)][int(r[2*d])]  #往程航班
        ret=flights[(destination,origin)][int(r[2*d+1])]  #返程航班
        print('%10s %10s %5s-%5s $%3s %5s-%5s $%3s' % (name,origin,out[0],out[1],out[2],ret[0],ret[1],ret[2]))



# 计算某个给定时间在一天中的分钟数
def getminutes(t):
    x = time.strptime(t, '%H:%M')
    return x[3] * 60 + x[4]

# 成本函数。输入为数字序列
def schedulecost(sol):
    totalprice=0
    latestarrival=0
    earliestdep=24*60
    for d in range(int(len(sol)/2)):
        # 得到往返航班
        origin=peoplelist[d][1]  #获取人员的来源地
        outbound=flights[(origin,destination)][int(sol[2*d])]  #获取往程航班
        returnf=flights[(destination,origin)][int(sol[2*d+1])]  #获取返程航班

        # 总价格等于所有往返航班的价格之和
        totalprice+=outbound[2]
        totalprice+=returnf[2]

        # 记录最晚到达和最早离开的时间
        if latestarrival<getminutes(outbound[1]): latestarrival=getminutes(outbound[1])
        if earliestdep>getminutes(returnf[0]): earliestdep=getminutes(returnf[0])

    # 接机服务：每个人必须在机场等待直到最后一个人到达位置
    # 送机服务：他们必须同时达到机场，并等待他们的返程航班
    totalwait=0
    for d in range(int(len(sol)/2)):
        origin=peoplelist[d][1]
        outbound=flights[(origin,destination)][int(sol[2*d])]
        returnf=flights[(destination,origin)][int(sol[2*d+1])]
        totalwait+=latestarrival-getminutes(outbound[1])
        totalwait+=getminutes(returnf[0])-earliestdep

        # 这个题解要求多付一天的汽车出租费用么？如果是，则费用为50美元
    if latestarrival>earliestdep: totalprice+=50

    return totalprice+totalwait

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




if __name__=="__main__":     #只有在执行当前模块时才会运行此函数
    print(flights)   #打印所有航班信息
    domain=[]
    for start_stop in flights:              #查询每个起止点的航班个数
        domain.append((0,len(flights[start_stop])-1))    #获取题解范围，两边必区间（航班范围的数据序列）

    # domain=[(0,9)]*(len(peoplelist)*2)
    print(domain)
    s=randomoptimize(domain,schedulecost)  # 随机搜索法，寻找最优题解
    print(s)
    printschedule(s)  #打印最优航班信息
    s = hillclimb(domain, schedulecost)  # 爬山法，寻找最优题解
    print(s)
    printschedule(s)  # 打印最优航班信息
    s = annealingoptimize(domain, schedulecost)  # 模拟退火算法，寻找最优题解
    print(s)
    printschedule(s)  # 打印最优航班信息
    s = geneticoptimize(domain, schedulecost)  # 遗传算法，寻找最优题解
    print(s)
    printschedule(s)  # 打印最优航班信息
