# 将优化算法应用到绘制关系网图中，使得交叉线最少

import math
import optimization

# 场景：在绘制关系网图中，每个人在图中都有位置x，y坐标，在有关联的两个人中间添加连线。
# 目标：是所有连线的交叉数最少

# 将题解转化为数字间没有联系的数字序列。可以让每个数字代表人物在图形中的位置x，y
# 例如：[200,120,250,230..]表示第1个人坐标为200,120，第2个人坐标为250,230...


# 关系网中涉及的人员
people=['Charlie','Augustus','Veruca','Violet','Mike','Joe','Willy','Miranda']

# 关联关系
links=[('Augustus', 'Willy'), 
       ('Mike', 'Joe'), 
       ('Miranda', 'Mike'), 
       ('Violet', 'Augustus'), 
       ('Miranda', 'Willy'), 
       ('Charlie', 'Mike'), 
       ('Veruca', 'Joe'), 
       ('Miranda', 'Augustus'), 
       ('Willy', 'Augustus'), 
       ('Joe', 'Charlie'), 
       ('Veruca', 'Augustus'), 
       ('Miranda', 'Joe')]

# 计算交叉线，作为成本函数
def crosscount(v):
  # 将数字序列转化为一个person:(x,y)的字典
  loc=dict([(people[i],(v[i*2],v[i*2+1])) for i in range(0,len(people))])
  total=0
  
  # 遍历每一对连线
  for i in range(len(links)):
    for j in range(i+1,len(links)):

      # 获取坐标位置
      (x1,y1),(x2,y2)=loc[links[i][0]],loc[links[i][1]]
      (x3,y3),(x4,y4)=loc[links[j][0]],loc[links[j][1]]
      
      den=(y4-y3)*(x2-x1)-(x4-x3)*(y2-y1)

      # 如果两线平行，则den==0。两条线是线段，不是直线
      if den==0: continue

      # 否则，ua与ub就是两条交叉线的分数值。
      # line where they cross
      ua=((x4-x3)*(y1-y3)-(y4-y3)*(x1-x3))/den
      ub=((x2-x1)*(y1-y3)-(y2-y1)*(x1-x3))/den
      
      # 如果两条线的分数值介于0和1之间，则两线彼此交叉
      if ua>0 and ua<1 and ub>0 and ub<1:
        total+=1
    for i in range(len(people)):
      for j in range(i+1,len(people)):
        # 获取两个节点的位置
        (x1,y1),(x2,y2)=loc[people[i]],loc[people[j]]

        # 获取两节点之间的距离
        dist=math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))
        # 对间距小于50个像素的节点进行判罚
        if dist<50:
          total+=(1.0-(dist/50.0))
        
  return total


#画图，绘制网络
from PIL import Image,ImageDraw

def drawnetwork(sol):
  # 建立image对象
  img=Image.new('RGB',(400,400),(255,255,255))
  draw=ImageDraw.Draw(img)

  # 建立标识位置信息的字典
  pos=dict([(people[i],(sol[i*2],sol[i*2+1])) for i in range(0,len(people))])

  for (a,b) in links:
    draw.line((pos[a],pos[b]),fill=(255,0,0))

  for n,p in pos.items():
    draw.text(p,n,(0,0,0))

  img.show()


domain=[(10,370)]*(len(people)*2)  #设定题解范围

if __name__=="__main__":     #只有在执行当前模块时才会运行此函数
    print(domain)
    s = optimization.randomoptimize(domain, crosscount)  # 随机搜索法，寻找最优题解
    print(s)
    drawnetwork(s)  # 绘制关系网
    s = optimization.hillclimb(domain, crosscount)  # 爬山法，寻找最优题解
    print(s)
    drawnetwork(s)  # 绘制关系网
    s = optimization.annealingoptimize(domain, crosscount)  # 模拟退火算法，寻找最优题解
    print(s)
    drawnetwork(s)  # 绘制关系网
    s = optimization.geneticoptimize(domain, crosscount)  # 遗传算法，寻找最优题解
    print(s)
    drawnetwork(s)  # 绘制关系网