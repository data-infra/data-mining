# 利用之前设计好的优化算法，解决涉及偏好的优化问题。这里解决学生住宿问题
import random
import math
import optimization

# 场景：每个房间有两个床位，每个学生有自己首选房间和次选房间（只选房间，不选床位）。将所有学生安排到所有房间
# 目标：在尽量满足学生的选择的情况下，为学生安排宿舍

# 将题解转化为数字间没有联系的数字序列。可以让每个数字代表可选床位的第几个，索引从0开始
# 例如：[0,2,1,1,1,2,0,1]表示第1个人选可选床位的第1个，第2个人选剩余可选床位的第3个床位，第3个人选剩余可选床位的第2个。。。

# 代表宿舍，每个宿舍有两个床位。5个房间
dorms=['Zeus','Athena','Hercules','Bacchus','Pluto']

# 代表学生及其首选房间和次选房间。10个人
prefs=[('Toby', ('Bacchus', 'Hercules')),
       ('Steve', ('Zeus', 'Pluto')),
       ('Karen', ('Athena', 'Zeus')),
       ('Sarah', ('Zeus', 'Pluto')),
       ('Dave', ('Athena', 'Bacchus')), 
       ('Jeff', ('Hercules', 'Pluto')), 
       ('Fred', ('Pluto', 'Athena')), 
       ('Suzie', ('Bacchus', 'Hercules')), 
       ('Laura', ('Bacchus', 'Hercules')), 
       ('James', ('Hercules', 'Athena'))]

# [(0,9),(0,8),(0,7),(0,6),...,(0,0)]   题解范围。因为前面选择了一个，后面的可选范围就会变少
domain=[(0,(len(dorms)*2)-i-1) for i in range(0,len(dorms)*2)]   #题解的可选范围


# 打印输出题解。输入为题解序列
def printsolution(vec):
  slots=[]
  # 为每个宿舍键两个槽
  for i in range(len(dorms)): slots+=[i,i]

  # 遍历每一名学生的安置情况
  for i in range(len(vec)):
    x=int(vec[i])

    # 从剩余槽中选择
    dorm=dorms[slots[x]]
    # 输出学生及其被分配的宿舍
    print(prefs[i][0],dorm)
    # 删除该槽，这样后面的数字列表才能正确翻译成“剩余床位”
    del slots[x]

# 成本函数：
def dormcost(vec):
  cost=0
  # 创建一个槽序列
  slots=[0,0,1,1,2,2,3,3,4,4]

  # 遍历每一名学生
  for i in range(len(vec)):
    x=int(vec[i])
    dorm=dorms[slots[x]]
    pref=prefs[i][1]
    # 首选成本值为0，次选成本值为1
    if pref[0]==dorm: cost+=0
    elif pref[1]==dorm: cost+=1
    else: cost+=3
    # 不在选择之列则成本值为3

    # 删除选中的槽
    del slots[x]
    
  return cost

if __name__=="__main__":     #只有在执行当前模块时才会运行此函数
    print(domain)
    s = optimization.randomoptimize(domain, dormcost)  # 随机搜索法，寻找最优题解
    print(s)
    printsolution(s)  # 打印最优解代表的含义
    s = optimization.hillclimb(domain, dormcost)  # 爬山法，寻找最优题解
    print(s)
    printsolution(s)  # 打印最优解代表的含义
    s = optimization.annealingoptimize(domain, dormcost)  # 模拟退火算法，寻找最优题解
    print(s)
    printsolution(s)  # 打印最优解代表的含义
    s = optimization.geneticoptimize(domain, dormcost)  # 遗传算法，寻找最优题解
    print(s)
    printsolution(s)  # 打印最优解代表的含义