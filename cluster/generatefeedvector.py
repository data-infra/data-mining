# -- coding: utf-8 --
#解析数据源，选择性提取特征数据集并存储。（特征数据集是用来聚类的）
import feedparser  #pip install feedparser
import re
import  urllib
# 提取特征数据集（链接-单词-次数）
# 返回一个RSS订阅源的标题和包含单词计数情况的字典
def getwordcounts(url):
    return
    # 解析订阅源
    d=feedparser.parse(url)
    wc={}

    # 循环遍历所有文章条目
    for e in d.entries:
        if 'summary' in e: summary=e.summary
        else: summary=e.description

        # 提取一个单词列表
        words=getwords(e.title+' '+summary)
        for word in words:
            wc.setdefault(word,0)
            wc[word]+=1
    return d.feed.title,wc   #返回输入变量和特征数据集

#根据源代码提取单词列表
def getwords(html):
    # 去除所有的html标记
    txt=re.compile(r'<[^>]+>').sub('',html)

    # 利用所有非字母字符拆分出单词
    words=re.compile(r'[^A-Z^a-z]+').split(txt)

    # 转化成小写形式
    return [word.lower() for word in words if word!='']


apcount={}    #第一个特征数据集：每个特征出现在的输入数目（每个单词出现在多少文章中）
wordcounts={} #第二个特征数据集：每个输入出现的特征数目（每篇文章包含的单词的数目）
feedlist=[line for line in open('feedlist.txt')]
for feedurl in feedlist:
    try:
        title,wc=getwordcounts(feedurl)
        print(title)
        print(wc)
        wordcounts[title]=wc
        for word,count in wc.items():
            apcount.setdefault(word,0)
            if count>1:
                apcount[word]+=1
    except:
        print('Failed to parse feed %s' % feedurl)

# 选取部分特征进行分析。因为特征出现次数太少具有偶然性，太多了具有普遍性，没法用于区分
wordlist=[]
for w,bc in apcount.items():
    frac=float(bc)/len(feedlist)
    if frac>0.1 and frac<0.5:
        wordlist.append(w)

# 将要分析的特征写入文件。最终形式为每行代表一个输入（文章），每列代表一个特征（单词），取值为出现的数量
out=open('blogdata1.txt','w')
out.write('Blog')
for word in wordlist: out.write('\t%s' % word)
out.write('\n')
# 将要分析的特征数据集写入文件
for blog,wc in wordcounts.items():
    print(blog)
    out.write(blog)
    for word in wordlist:
        if word in wc: out.write('\t%d' % wc[word])
        else: out.write('\t0')
    out.write('\n')

