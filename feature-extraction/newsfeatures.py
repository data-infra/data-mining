# 非监督式的特征提取。寻找独立特征（主题）。这些特征（主题）可能不能简单的描述
# 如果用于分类，不用提前知道需要什么分类，也不需要数据训练
# 对于文章分类，特征（主题）可能不能简单的描述，但是可以计算每个属性（单词）对特征（主题）的重要程度

# 注意和分类器中区分，在分类器中我们把单词作为文章的特征，更缺的说的属性特征，这里我们要研究的特征是输入对象表现出来的主题特征，单词作为研究主题特征的参考属性。

import feedparser
import re


feedlist=['http://today.reuters.com/rss/topNews',
          'http://today.reuters.com/rss/domesticNews',
          'http://today.reuters.com/rss/worldNews',
          'http://hosted.ap.org/lineups/TOPHEADS-rss_2.0.xml',
          'http://hosted.ap.org/lineups/USHEADS-rss_2.0.xml',
          'http://hosted.ap.org/lineups/WORLDHEADS-rss_2.0.xml',
          'http://hosted.ap.org/lineups/POLITICSHEADS-rss_2.0.xml',
          'http://www.nytimes.com/services/xml/rss/nyt/HomePage.xml',
          'http://www.nytimes.com/services/xml/rss/nyt/International.xml',
          'http://news.google.com/?output=rss',
          'http://feeds.salon.com/salon/news',
          'http://www.foxnews.com/xmlfeed/rss/0,4313,0,00.rss',
          'http://www.foxnews.com/xmlfeed/rss/0,4313,80,00.rss',
          'http://www.foxnews.com/xmlfeed/rss/0,4313,81,00.rss',
          'http://rss.cnn.com/rss/edition.rss',
          'http://rss.cnn.com/rss/edition_world.rss',
          'http://rss.cnn.com/rss/edition_us.rss']

# 删除文章中所有的图片和HTML标记
def stripHTML(h):
    p=''
    s=0
    for c in h:
        if c=='<': s=1
        elif c=='>':
            s=0
            p+=' '
        elif s==0: p+=c
    return p

# 拆分文章中的单词
def separatewords(text):
    splitter=re.compile('\\W*')
    return [s.lower() for s in splitter.split(text) if len(s)>3]

#获取文章的单词。单词出现一次就记录一次。
def getarticlewords():
    allwords={}
    articlewords=[]
    articletitles=[]
    ec=0
    # 遍历每个订阅源
    for feed in feedlist:
        f=feedparser.parse(feed)

        # 遍历每篇文章
        for e in f.entries:
            # 跳过标题相同的文章
            if e.title in articletitles: continue

            # 提取单词
            txt=e.title.encode('utf8')+stripHTML(e.description.encode('utf8'))
            words=separatewords(txt)
            articlewords.append({})
            articletitles.append(e.title)

            # 在allwords和articlewords中增加针对当前单词的计数
            for word in words:
                allwords.setdefault(word,0)
                allwords[word]+=1
                articlewords[ec].setdefault(word,0)
                articlewords[ec][word]+=1   #单词出现一次就记录一次。
            ec+=1
    # allwords记录单词在所有文章中被使用的次数，我们将利用该变量来判断，哪些单词应该被视作特征的一部分
    # articlewords是单词在每篇文章中出现的次数
    # articletitles是一个文章标题的列表
    return allwords,articlewords,articletitles


# 将数据集转化为矩阵。allw为单词在所有文章中出现的次数，articlew为单词在每篇文章中出现的次数
def makematrix(allw,articlew):
    wordvec=[]

    # 只考虑那些普通的但又不至于非常普通的单词
    for w,c in allw.items():
        if c>3 and c<len(articlew)*0.6:
            wordvec.append(w)

    # 构建单词矩阵
    l1=[[(word in f and f[word] or 0) for word in wordvec] for f in articlew]
    return l1,wordvec

from numpy import *


# 显示特征（主题），以单词和单词权重的形式列出特征（主题）
def showfeatures(w,h,titles,wordvec,out='features.txt'):
    outfile=open(out,'w')
    pc,wc=shape(h)
    toppatterns=[[] for i in range(len(titles))]
    patternnames=[]

    # 遍历所有特征
    for i in range(pc):
        slist=[]
        # 构造一个包含单词及其权重数据的列表
        for j in range(wc):
            slist.append((h[i,j],wordvec[j]))
        # 将单词列表倒序排列
        slist.sort()
        slist.reverse()

        # 打印开始的6个元素
        n=[s[1] for s in slist[0:6]]
        outfile.write(str(n)+'\n')
        patternnames.append(n)

        # 构造一个针对该特征的文章列表
        flist=[]
        for j in range(len(titles)):
            # 加入文章及其权重数据
            flist.append((w[j,i],titles[j]))
            toppatterns[j].append((w[j,i],i,titles[j]))

        # 将该列表倒序排列
        flist.sort()
        flist.reverse()

        # 显示前3篇文章
        for f in flist[0:3]:
            outfile.write(str(f)+'\n')
        outfile.write('\n')

    outfile.close()
    # 返回模式名称，以供后续使用
    return toppatterns,patternnames

# 以文章和文章权重的形式列出特征（主题）
def showarticles(titles,toppatterns,patternnames,out='articles.txt'):
    outfile=open(out,'w')

    # 遍历所有文章
    for j in range(len(titles)):
        outfile.write(titles[j].encode('utf8')+'\n')

        # 针对该篇文章，获得排位最靠前的几个特征，并将其按倒序排列
        toppatterns[j].sort()
        toppatterns[j].reverse()

        # 打印前3个模式
        for i in range(3):
            outfile.write(str(toppatterns[j][i][0])+' '+ str(patternnames[toppatterns[j][i][1]])+'\n')
        outfile.write('\n')

    outfile.close()
