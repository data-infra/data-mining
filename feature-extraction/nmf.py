# 算法实现

from numpy import *

# 误差成本，对最终结果与理想结果的接近程度，加以衡量
def difcost(a,b):
  dif=0
  # 遍历矩阵中的每一行和每一列
  for i in range(shape(a)[0]):
    for j in range(shape(a)[1]):
      # 将差值相加
      dif+=pow(a[i,j]-b[i,j],2)
  return dif
# 搜索最优解，使用乘法更新法则。逐步更新矩阵，使成本函数的计算值逐步降低。需要搜索确定分类的数目。当然可以通过实验手段先找到一个合理的范围，再确定分类数目
def factorize(v,pc=10,iter=50):
  ic=shape(v)[0]
  fc=shape(v)[1]

  # 以随机值初始化权重矩阵和特征矩阵
  w=matrix([[random.random() for j in range(pc)] for i in range(ic)])
  h=matrix([[random.random() for i in range(fc)] for i in range(pc)])

  # 最多执行iter次操作
  for i in range(iter):
    wh=w*h
    
    # 计算当前差值
    cost=difcost(v,wh)
    
    if i%10==0: print(cost)
    
    # 如果矩阵已经分解彻底，则立即终止
    if cost==0: break
    
    # 更新特征矩阵
    hn=(transpose(w)*v)
    hd=(transpose(w)*w*h)
  
    h=matrix(array(h)*array(hn)/array(hd))

    # 更新权重矩阵
    wn=(v*transpose(h))
    wd=(w*h*transpose(h))

    w=matrix(array(w)*array(wn)/array(wd))  
    
  return w,h
