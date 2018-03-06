# 股票成交量
import nmf
import urllib2
from numpy import *

tickers=['YHOO','AVP','BIIB','BP','CL','CVX',
         'DNA','EXPE','GOOG','PG','XOM','AMGN']

shortest=300
prices={}
dates=None

for t in tickers:
  # 打开url
  rows=urllib2.urlopen('http://ichart.finance.yahoo.com/table.csv?'+\
                       's=%s&d=11&e=26&f=2006&g=d&a=3&b=12&c=1996'%t +\
                       '&ignore=.csv').readlines()

  
  # 从每一行中提取成交量
  prices[t]=[float(r.split(',')[5]) for r in rows[1:] if r.strip()!='']
  if len(prices[t])<shortest: shortest=len(prices[t])
  
  if not dates:
    dates=[r.split(',')[0] for r in rows[1:] if r.strip()!='']

# 一行代表一个输入对象（成交日）内一批属性（股票）的取值（成交量）
l1=[[prices[tickers[i]][j] for i in range(len(tickers))] for j in range(shortest)]

w,h=nmf.factorize(matrix(l1), pc=5)

print(h)
print(w)

# 遍历所有特征
for i in range(shape(h)[0]):
  print("Feature %d" % i)
  
  # 得到最符合当前特征的股票
  ol=[(h[i,j],tickers[j]) for j in range(shape(h)[1])]
  ol.sort()
  ol.reverse()
  for j in range(12):
    print(ol[j])
  print
  
  # 显示最符合当前特征的交易日期
  porder=[(w[d,i],d) for d in range(300)]
  porder.sort()
  porder.reverse()
  print([(p[0],dates[p[1]]) for p in porder[0:3]])
  print
