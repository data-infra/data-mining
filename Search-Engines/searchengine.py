# 搜索和排名
import urllib
from bs4 import BeautifulSoup
import re
import sqlite3
import nn
import os
import spyder   #获取爬虫数据集

# 分词时忽略下列词
biaodian = '[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+ ，。？‘’“”！；：\r\n、（）…'     #所有的标点符号
# ignorewords=['，','。','？','“','”','！','；','：','\n','、','-',',','.','?','\r\n','_',' ']
ignorewords = list(set(biaodian))   #去重
ignorewords.append('\r\n')   #添加一个不忽略的项



# 定义搜索引擎类
class searcher:
    def __init__(self,dbname):
        self.con=sqlite3.connect(dbname)  #链接数据库
        self.curs = self.con.cursor()

    def __del__(self):
        self.curs.close()
        self.con.close()


    # 根据搜索字符串分词后获取查询到的链接
    def getmatchrows(self,querystr):
        # 构造数据库的查询字符串（搜索字符串根据空格分割成查询字符串列表）
        fieldlist='w0.urlid'
        tablelist=''
        clauselist=''
        wordids=[]

        # 根据空格分割单词
        words=querystr.strip().split(' ')
        tablenumber=0
        for word in words:
            # 获取单词的id
            wordrow=self.curs.execute("select rowid from wordlist where word='%s'" % word).fetchall()
            if wordrow!=None and len(wordrow)> 0:
                wordid=wordrow[0][0]  #获取单词id
                wordids.append(wordid)
                if tablenumber>0:
                    tablelist+=','
                    clauselist+=' and '
                    clauselist+='w%d.urlid=w%d.urlid and ' % (tablenumber-1,tablenumber)
                fieldlist+=',w%d.location' % tablenumber
                tablelist+='wordlocation w%d' % tablenumber
                clauselist+='w%d.wordid=%d' % (tablenumber,wordid)
                tablenumber+=1

        # 根据各个组分，建立查询。为列表中的每个单词，建立指向wordlocation表的引用，并根据对应的urlid将它们连接起来进行联合查询
        fullquery='select %s from %s where %s' % (fieldlist,tablelist,clauselist)
        # print(fullquery)
        cur=self.curs.execute(fullquery)
        rows=[row for row in cur.fetchall()]

        return rows,wordids

    # 对查询到的链接进行排名。参数：rows，wordids查询字符串id列表
    def getscoredlist(self,rows,wordids):
        totalscores=dict([(row[0],0) for row in rows])
        # 对链接进行评价的函数。（权重和评价值），使用了多种评价函数
        weights=[(1.0,self.locationscore(rows)),   #根据关键词出现的位置获取权重
                 (1.0,self.frequencyscore(rows)),  #根据关键词出现的频率获取权重
                 (1.0,self.pagerankscore(rows)),   #根据pagerank获取权重
                 (1.0,self.linktextscore(rows,wordids)), #根据链接描述获取权重
                 (5.0,self.nnscore(rows,wordids))]  #根据神经网络获取权重
        for (weight,scores) in weights:
            for urlid in totalscores:
                totalscores[urlid]+=weight*scores[urlid]

        return totalscores  #返回每个链接的评价值

    #根据urlid查询url
    def geturlname(self,id):
        return self.curs.execute("select url from urllist where rowid=%d" % id).fetchall()[0][0]

    #搜索函数：将上面的搜索、评价、排名合并在一起
    def query(self,querystr):
        rows,wordids=self.getmatchrows(querystr)  #rows是[urlid,wordlocation1,wordlocation2,wordlocation3...]
        if rows==None or len(rows)==0:
            print('无法查询到,请使用空格分隔查询关键词')
            return

        scores=self.getscoredlist(rows,wordids)
        rankedscores=[(score,url) for (url,score) in scores.items()]
        rankedscores.sort()
        rankedscores.reverse()
        for (score,urlid) in rankedscores[0:10]:
            print('%f\t%d\t%s' % (score,urlid,self.geturlname(urlid)))
        return wordids,[r[1] for r in rankedscores[0:10]]


    # 评价值归一化：因为不同的评价方法的返回值和含义不同。这里所有的评价值归一化到0-1,默认越大越好
    def normalizescores(self,scores,smallIsBetter=0):
        vsmall=0.00001 #避免被0整除
        if smallIsBetter:
            minscore=min(scores.values())
            return dict([(u,float(minscore)/max(vsmall,l)) for (u,l) in scores.items()])
        else:
            maxscore=max(scores.values())
            if maxscore==0: maxscore=vsmall
            return dict([(u,float(c)/maxscore) for (u,c) in scores.items()])

    # 根据单词频度进行评价的函数.#rows是[urlid,wordlocation1,wordlocation2,wordlocation3...]
    def frequencyscore(self,rows):
        counts=dict([(row[0],0) for row in rows])
        for row in rows: counts[row[0]]+=1
        return self.normalizescores(counts)

    # 根据单词位置进行评价的函数.#rows是[urlid,wordlocation1,wordlocation2,wordlocation3...]
    def locationscore(self,rows):
        locations=dict([(row[0],1000000) for row in rows])
        for row in rows:
            loc=sum(row[1:])
            if loc<locations[row[0]]: locations[row[0]]=loc

        return self.normalizescores(locations,smallIsBetter=1)

    # 根据单词距离进行评价的函数。#rows是[urlid,wordlocation1,wordlocation2,wordlocation3...]
    def distancescore(self,rows):
        # 如果仅查询了一个单词，则得分都一样
        if len(rows[0])<=2: return dict([(row[0],1.0) for row in rows])

        # 初始化字典，并填入一个很大的值
        mindistance=dict([(row[0],1000000) for row in rows])

        for row in rows:
            dist=sum([abs(row[i]-row[i-1]) for i in range(2,len(row))])
            if dist<mindistance[row[0]]: mindistance[row[0]]=dist
        return self.normalizescores(mindistance,smallIsBetter=1)

    # 利用外部回值链接进行评价（仅计算回指数目）。#rows是[urlid,wordlocation1,wordlocation2,wordlocation3...]
    def inboundlinkscore(self,rows):
        uniqueurls=dict([(row[0],1) for row in rows])
        inboundcount=dict([(u,self.curs.execute('select count(*) from link where toid=%d' % u).fetchall()[0]) for u in uniqueurls])
        return self.normalizescores(inboundcount)

    # 利用链接文本进行评价的函数。#rows是[urlid,wordlocation1,wordlocation2,wordlocation3...]
    def linktextscore(self,rows,wordids):
        linkscores=dict([(row[0],0) for row in rows])
        for wordid in wordids:
            cur=self.curs.execute('select link.fromid,link.toid from linkwords,link where wordid=%d and linkwords.linkid=link.rowid' % wordid)
            for (fromid,toid) in cur.fetchall():
                if toid in linkscores:
                    pr=self.curs.execute('select score from pagerank where urlid=%d' % fromid).fetchall()[0][0]
                    linkscores[toid]+=pr
        maxscore=max(linkscores.values())   #求最大的pagerank值
        for urlid in linkscores:
            linkscores[urlid] /= maxscore   #归一化
        return linkscores

    # 根据pagerank值进行评价的函数。（利用外部回值链接进行评价）。#rows是[urlid,wordlocation1,wordlocation2,wordlocation3...]
    def pagerankscore(self,rows):
        pageranks=dict([(row[0],self.curs.execute('select score from pagerank where urlid=%d' % row[0]).fetchall()[0][0]) for row in rows])
        maxrank=max(pageranks.values())   #求最大的pagerank值
        for urlid in pageranks:
            pageranks[urlid] /= maxrank   #归一化
        return pageranks   #返回归一化的url的pagerank

    # 根据神经网络（用户点击行为学习）进行评价的函数。神经网络在nn.py中实现。#rows是[urlid,wordlocation1,wordlocation2,wordlocation3...]
    def nnscore(self,rows,wordids):
        # 获得一个由唯一的url id构成的有序列表
        urlids=[urlid for urlid in dict([(row[0],1) for row in rows])]
        nnres=mynet.getresult(wordids,urlids)
        scores=dict([(urlids[i],nnres[i]) for i in range(len(urlids))])
        return self.normalizescores(scores)


mynet=nn.searchnet('csdn.db')
if __name__ == '__main__':
    mysearcher= searcher('csdn.db')
    searchkey = input("搜索关键词>")
    wordids,urlids=mysearcher.query(searchkey)
    # print(wordids,urlids)
    selurlid= input("选中链接id>")
    selurlid = int(selurlid)
    mynet.trainquery(wordids, urlids,selurlid) #根据用户选择的链接进行训练








