# 神经网络学习用户点击行为，实现搜索排名
from math import tanh
import sqlite3

# 指定任何输出值的斜率
def dtanh(y):
    return 1.0-y*y

class searchnet:
    def __init__(self,dbname):
        self.con=sqlite3.connect(dbname)  #链接数据库
        self.curs = self.con.cursor()
        try:
            self.maketables()
        except:
            pass

    def __del__(self):
        self.curs.close()
        self.con.close()

    def dbcommit(self):  # 保存数据库
        self.con.commit()

    def maketables(self):
        self.curs.execute('create table hiddennode(create_key)')              #创建神经元节点
        self.curs.execute('create table wordhidden(fromid,toid,strength)')   #输入节点（单词）到神经元的权重，默认-0.2
        self.curs.execute('create table hiddenurl(fromid,toid,strength)')    #神经元到输出节点（链接）的权重，默认为0
        self.dbcommit()

        # 获取节点间权重。layer为0表示输入到神经元之间，为1表示神经元到输出之间
    def getstrength(self,fromid,toid,layer):
        if layer==0: table='wordhidden'
        else: table='hiddenurl'
        res=self.curs.execute('select strength from %s where fromid=%d and toid=%d' % (table,fromid,toid)).fetchall()
        if res==None or len(res)==0:
            if layer==0: return -0.2  # 输入到神经元之间默认为-0.2
            if layer==1: return 0   # 神经元到输出节点间默认为0
        return res[0][0]

    # 设置数据库中节点间权重。layer为0表示输入到神经元之间，为1表示神经元到输出之间
    def setstrength(self,fromid,toid,layer,strength):
        if layer==0: table='wordhidden'
        else: table='hiddenurl'
        res=self.curs.execute('select rowid from %s where fromid=%d and toid=%d' % (table,fromid,toid)).fetchall()
        if res==None or len(res)==0:
            self.curs.execute('insert into %s (fromid,toid,strength) values (%d,%d,%f)' % (table,fromid,toid,strength))
        else:
            rowid=res[0][0]
            self.curs.execute('update %s set strength=%f where rowid=%d' % (table,strength,rowid))

     # 根据已知正确的输入输出结果，创建神经元,建立输入与神经元间里连接，神经元与输出间连接。每一种输入组合都创建一个神经节点
    def generatehiddennode(self,wordids,urlids):
        # if len(wordids)>3: return None   #对于有3个输出单词的我们不做处理，太复杂
        # 检测我们是否已经为这组输入建好了一个节点
        sorted_words=[str(id) for id in wordids]
        sorted_words.sort()
        createkey='_'.join(sorted_words)
        res=self.curs.execute("select rowid from hiddennode where create_key='%s'" % createkey).fetchall()

        # 如果没有，就创建
        if res==None or len(res)==0:
            cur=self.curs.execute("insert into hiddennode (create_key) values ('%s')" % createkey)
            hiddenid=cur.lastrowid
            # 设置默认权重
            for wordid in wordids:
                self.setstrength(wordid,hiddenid,0,1.0/len(wordids))   #设置输入节点到神经元间的权重为1/输入数量
            for urlid in urlids:
                self.setstrength(hiddenid,urlid,1,0.1)   #设置神经元到输出节点间权重为0.1
            self.dbcommit()

    # 根据输入关键词id和相关的链接的id，获取数据库中神经元节点的id。（相关的链接也就是初步查询到的链接，只要链接的网页中出现过关键词就会被认为相关）
    def getallhiddenids(self,wordids,urlids):
        hiddenids=[]
        for wordid in wordids:
            cur=self.curs.execute('select toid from wordhidden where fromid=%d' % wordid).fetchall()
            for row in cur:
                hiddenids.append(row[0])
        for urlid in urlids:
            cur=self.curs.execute('select fromid from hiddenurl where toid=%d' % urlid).fetchall()
            for row in cur:
                hiddenids.append(row[0])
        return hiddenids  # 返回神经元节点

    # 构建一个神经网络
    def setupnetwork(self,wordids,urlids):
        # 值列表：输入：神经元、输出
        self.wordids=wordids
        self.hiddenids=self.getallhiddenids(wordids,urlids)
        self.urlids=urlids

        # 构建输入节点、神经元、输出节点。就是前面的输入、神经元、输出。这里用了一个更加普遍的名称
        self.ai = [1.0]*len(self.wordids)
        self.ah = [1.0]*len(self.hiddenids)
        self.ao = [1.0]*len(self.urlids)

        # 建立权重矩阵（线性组合系数矩阵）：输入-神经元，  神经元-输出
        self.wi = [[self.getstrength(wordid,hiddenid,0)
                    for hiddenid in self.hiddenids]
                    for wordid in self.wordids]
        self.wo = [[self.getstrength(hiddenid,urlid,1)
                    for urlid in self.urlids]
                    for hiddenid in self.hiddenids]

    # 前馈算法：一列输入，进入神经网络，返回所有输出结果的活跃程度。越活跃也好。因为神经网络每向下传播一层就会衰弱一层。衰弱函数使用tanh这种0时陡峭，无限大或无限小时平稳的函数
    def feedforward(self):
        # 查询的单词是仅有的输入
        for i in range(len(self.wordids)):
            self.ai[i] = 1.0   #输入节点的活跃程度就设为1

        # 根据输入节点活跃程度，获取神经元节点的活跃程度
        for j in range(len(self.hiddenids)):
            sum = 0.0
            for i in range(len(self.wordids)):
                sum = sum + self.ai[i] * self.wi[i][j]  # 线性组合
            self.ah[j] = tanh(sum)   #使用tanh表示神经元对输入的反应强度。（因为tanh是一个在0附近震荡强烈，在远离0时趋于稳定的函数）

        # 根据神经元节点活跃程度，获取输出节点的活跃程度
        for k in range(len(self.urlids)):
            sum = 0.0
            for j in range(len(self.hiddenids)):
                sum = sum + self.ah[j] * self.wo[j][k]  # 线性组合
            self.ao[k] = tanh(sum)

        return self.ao[:]

# =============================以上是公共实例函数=======================================

#=======================getresult是应用神经网络进行搜索的函数==============================

    #  针对一组单词和url给出输出
    def getresult(self,wordids,urlids):
        self.setupnetwork(wordids,urlids)
        return self.feedforward()




# ============================下面是使用反向传播法进行神经网络训练============================


 #前馈训练法：依据当前权重预测输出，计算误差，更正权重。用户每选择一次，进行一次训练

  #用户每选择一次链接，就调整一次权重。targets表示正确的输出结果。即用户选择的链接
    def backPropagate(self, targets, N=0.5):
        # 计算输出层误差
        output_deltas = [0.0] * len(self.urlids)
        for k in range(len(self.urlids)):
            error = targets[k]-self.ao[k]    #计算正确输出和预测输出之间的误差
            output_deltas[k] = dtanh(self.ao[k]) * error   #确定总输入需要如何改变

        # 计算神经元误差：
        hidden_deltas = [0.0] * len(self.hiddenids)
        for j in range(len(self.hiddenids)):
            error = 0.0
            for k in range(len(self.urlids)):
                error = error + output_deltas[k]*self.wo[j][k]  #将每个神经元-输出间的权重值乘以输出节点的改变量，再累加求和，从而改变节点的输出结果
            hidden_deltas[j] = dtanh(self.ah[j]) * error  #确定节点的总输入所需的该变量

        # 更新神经元-输出间权重
        for j in range(len(self.hiddenids)):
            for k in range(len(self.urlids)):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change

        # 更新输入-神经元间权重
        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change


    # 更新数据库
    def updatedatabase(self):
        # 将值写入数据库
        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                self.setstrength(self.wordids[i],self. hiddenids[j],0,self.wi[i][j])
        for j in range(len(self.hiddenids)):
            for k in range(len(self.urlids)):
                self.setstrength(self.hiddenids[j],self.urlids[k],1,self.wo[j][k])
        self.dbcommit()


   # 对神经网络进行一次训练。wordids为查询关键词id，urlids为查找到相关的url的id，selectedurl为选中的url的id
    def trainquery(self,wordids,urlids,selectedurlid):
        # 第一次运行时，在数据库中创建表。以默认权重赋值
        self.generatehiddennode(wordids,urlids)
        # 创建神经网络类的属性。（输入、神经元、输出、两个权重）
        self.setupnetwork(wordids,urlids)
        self.feedforward()   #执行前馈算法，根据输入获取输出
        targets=[0.0]*len(urlids)
        targets[urlids.index(selectedurlid)]=1.0  #获取用户选择的正确链接
        error = self.backPropagate(targets)  #执行反向传播法修正网络
        self.updatedatabase()   #更新数据库


if __name__ == '__main__':
    con = sqlite3.connect('csdn.db')
    curs = con.cursor()
    command = 'select * from hiddennode'  # "select fromid,toid from link"
    cur = curs.execute(command)
    res = cur.fetchall()
    if res != None and len(res) > 0:
        for row in res:
            print(row)



