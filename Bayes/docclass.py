# 文档过滤，垃圾邮件分类。跟聚类(不知道多少分类，也不知道有什么类别)不同，它是一种监督式的学习方法。因此也就在分类前已知道分类（垃圾邮件和非垃圾邮件）。判断对象属于哪种分类。
import sqlite3
import re
import math


# 邮件数据集，，邮件内容和所属分类
data=[
    ['Nobody owns the water.','good'],
    ['the quick rabbit jumps fences','good'],
    ['buy pharmaceuticals now','bad'],
    ['make quick money at the online casino','bad'],
    ['the quick brown fox jumps','good']
]

# 从输入对象中提取特征（文档中提取不重复单词）
def getwords(doc):
    splitter=re.compile('\\W*')
    # print(doc)
    # 根据非字母字符进行单词拆分
    words=[s.lower() for s in splitter.split(doc) if len(s)>2 and len(s)<20]

    # 只返回一组不重复的单词。（特征不重复）
    return dict([(w,1) for w in words])


# 统计基类。主要为了计算特征概率Pr （word | classification ）。有多种不同的分类器方式，在此类上派生。定义成类，这样每个实例对象可以训练自己的数据集
class classifier:
    #getfeatures为特征提取函数
    def __init__(self,getfeatures,filename=None):
        # 统计特征/分类组合的数量。（每个特征在不同分类中的数量）
        self.fc={}
        # 统计每个分类中的文档数量。（每个分类中的输入对象的数量）
        self.cc={}
        self.getfeatures=getfeatures
        self.setdb('test.db')    #设定数据库

    # 链接数据库，创建表
    def setdb(self,dbfile):
        self.con=sqlite3.connect(dbfile)
        self.curs = self.con.cursor()
        self.curs.execute('create table if not exists fc(feature,category,count)')
        self.curs.execute('create table if not exists cc(category,count)')

    # 增加对特征feature/分类cat组合的计数值
    def incf(self,feature,cat):
        count=self.fcount(feature,cat)   #计算某一特征在某一分类中出现的次数
        if count==0:
            self.curs.execute("insert into fc values ('%s','%s',1)" % (feature,cat))
        else:
            self.curs.execute("update fc set count=%d where feature='%s' and category='%s'" % (count+1,feature,cat))

    # 查询某一特征出现于某一分类中的次数
    def fcount(self,feature,cat):
        res=self.curs.execute('select count from fc where feature="%s" and category="%s"' %(feature,cat)).fetchall()
        if res==None or len(res)==0: return 0
        else: return float(res[0][0])

    # 增加对某一分类的计数值
    def incc(self,cat):
        count=self.catcount(cat)
        if count==0:
            self.curs.execute("insert into cc values ('%s',1)" % (cat))
        else:
            self.curs.execute("update cc set count=%d where category='%s'" % (count+1,cat))

    # 查询属于某一分类的输入对象（文章）的数量
    def catcount(self,cat):
        res=self.curs.execute('select count from cc where category="%s"' %(cat)).fetchall()
        if res==None or len(res)==0: return 0
        else: return float(res[0][0])

    # 查询所有分类的列表。因为要计算每个属于每个分类的比例
    def categories(self):
        cur=self.curs.execute('select category from cc')
        return [d[0] for d in cur]

    # 所有输入对象（文章）的数量.这是一项无用的计算。
    def totalcount(self):
        res=self.curs.execute('select sum(count) from cc').fetchall()
        if res==None or len(res)==0: return 0
        return res[0][0]

    # 对样本进行统计训练，完善数据库。参数为：输入对象，所属分类
    def train(self,input,cat):
        features=self.getfeatures(input)  #提取特征
        # 针对该分类为提取到的特征增加计数值
        for feature in features:
            self.incf(feature,cat)

        # 增加针对该分类的计数值
        self.incc(cat)
        self.con.commit()

#=================上面是存储特征和分类=================
#=================下面是计算特征概率Pr （word | classification ）=================


    # 统计指定分类中某一特征出现的概率（单词在分类中出现的概率,Pr(word/classification)）
    def fprob(self,feature,cat):
        if self.catcount(cat)==0: return 0

        # 该特征在分类中出现的次数，除以分类中所有特征的数目
        return self.fcount(feature,cat)/self.catcount(cat)

    # 计算加权概率，为特征设置权重，避免极少特征的强烈震荡。比如money单词只出现了一次在垃圾邮件中。也就是100%是垃圾邮件。所以添加权重概率避免这种事情。
    def weightedprob(self,feature,cat,prf,weight=1.0,ap=0.5):
        # 计算在某一分类中某一特征出现的概率
        basicprob=prf(feature,cat)

        # 统计某一特征在所有分类中出现的次数
        totals=sum([self.fcount(feature,cat) for cat in self.categories()])

        # 计算加权平均
        bp=((weight*ap)+(totals*basicprob))/(weight+totals)
        return bp  #返回加权平均概率



# 派生朴素贝叶斯分类器（适用于特征间相互独立的情况）。将单词在目标分类中出现的概率相乘组成输入对象（文章）出现在分类中的概率。
class naivebayes(classifier):

    def __init__(self,getfeatures):
        classifier.__init__(self,getfeatures)
        self.thresholds={}

    # 获取输入对象（文章）出现在指定分类中的概率。Pr(Document|Categrory)
    def docprob(self,input,cat):
        features=self.getfeatures(input)
        # 将所有特征的概率相乘
        p=1
        for feature in features: p*=self.weightedprob(feature,cat,self.fprob)
        return p

    # 统计分类的概率，并返回Pr(Document|Category)*Pr(Category)/Pr(Document)。朴素贝叶斯用这个概率代表最终概率进行比较
    def prob(self,input,cat):
        catprob=self.catcount(cat)/self.totalcount()
        docprob=self.docprob(input,cat)   #获取指定分类中，输入对象的概率.Pr(Document|Categrory)
        return docprob*catprob

    def setthreshold(self,cat,t):
        self.thresholds[cat]=t

    def getthreshold(self,cat):
        if cat not in self.thresholds: return 1.0
        return self.thresholds[cat]

    def classify(self,input,default=None):
        probs={}
        # 寻找概率最大的分类
        max=0.0
        cats = self.categories()
        for cat in cats:
            probs[cat]=self.prob(input,cat)
            if probs[cat]>max:
                max=probs[cat]
                best=cat

        # 确保概率值超过阈值*次大概率值
        for cat in probs:
            if cat==best: continue
            if probs[cat]*self.getthreshold(best)>probs[best]: return default
        return best


# 派生费舍尔分类器。先计算某特征属于指定分类的概率Pr(Category|feature),再根据此概率计算费舍尔方法的值（所有概率相乘，取自然对数，再乘以-2）。
class fisherclassifier(classifier):
    # 计算某特征属于指定分类的概率Pr(Category|feature)
    def cprob(self,f,cat):
        # 特征在该分类中出现的概率
        clf=self.fprob(f,cat)
        if clf==0: return 0

        # 特征在所有分类中出现的频率
        freqsum=sum([self.fprob(f,c) for c in self.categories()])

        # 概率等于特征在该分类中出现的频率除以总体频率
        p=clf/(freqsum)

        return p

    # 根据输入对象获取属性特征，计算特征属于分类的概率，再计算费舍尔的值作为输入对象（文章）属于指定分类的概率
    def fisherprob(self,input,cat):
        # 将所有概率值相乘
        p=1
        features=self.getfeatures(input)
        for f in features:
            p*=(self.weightedprob(f,cat,self.cprob))

        # 取自然对数，并乘以-2
        fscore=-2*math.log(p)

        # 利用倒置对数卡方函数求得概率
        return self.invchi2(fscore,len(features)*2)
    # 倒置对数卡方函数
    def invchi2(self,chi, df):
        m = chi / 2.0
        sum = term = math.exp(-m)
        for i in range(1, df//2):
            term *= m / i
            sum += term
        return min(sum, 1.0)

    def __init__(self,getfeatures):
        classifier.__init__(self,getfeatures)
        self.minimums={}



    def setminimum(self,cat,min):
        self.minimums[cat]=min
    def getminimum(self,cat):
        if cat not in self.minimums: return 0
        return self.minimums[cat]

    # 使用具体的下限费舍尔值来对输入对象进行分类
    def classify(self,input,default=None):
        # 循环遍历并寻找最佳结果
        best=default
        max=0.0
        for c in self.categories():
            p=self.fisherprob(input,c)   #计算输入对象的费舍尔值
            # 确保其超过下限值
            if p>self.getminimum(c) and p>max:
                best=c
                max=p
        return best


# 简单的样本训练
def sampletrain(cl,data):
    for item in data:
        cl.train(item[0],item[1])


if __name__=="__main__":     #只有在执行当前模块时才会运行此函数
    cl=naivebayes(getwords)   #定义朴素贝叶斯分类器
    sampletrain(cl,data)      #训练样本数据（数据清洗、转换、提取）
    best = cl.classify('quick money')  #利用分类器进行分类
    print(best)               #打印分类结果

    cl = fisherclassifier(getwords)  # 定义费舍尔分类器
    sampletrain(cl, data)  # 训练样本数据（数据清洗、转换、提取）
    best = cl.classify('quick money')  # 利用分类器进行分类
    print(best)  # 打印分类结果









