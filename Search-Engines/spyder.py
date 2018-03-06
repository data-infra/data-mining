# 根据连接爬取中文网站，获取标题、子连接、子连接数目、连接描述、中文分词列表，
import urllib
from bs4 import BeautifulSoup
import bs4
import sqlite3
import os
import jieba   #对中文进行分词
import traceback

# 分词时忽略下列词
biaodian = '[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+ ，。？‘’“”！；：\r\n、（）…'     #所有的标点符号
ignorewords = list(set(biaodian))   #去重
ignorewords.append('\r\n')   #添加一个不忽略的项


# 定义爬虫类。获取链接的题目、描述、分词、深度
class crawler:
    def __init__(self,dbname):
        self.urls={}              #创建爬取列表
        self.con = sqlite3.connect(dbname)
        self.curs = self.con.cursor()
        try:
            self.createindextables()
        except:
            pass

    def __del__(self):  # 关闭数据库
        self.curs.close()
        self.con.close()

    def dbcommit(self):  # 保存数据库
        self.con.commit()

    # 创建数据库表
    def createindextables(self):
        self.curs.execute('create table urllist(url,hascrawl)')  # 创建url列表数据表
        self.curs.execute('create table wordlist(word)')  # 创建word列表数据表
        self.curs.execute('create table wordlocation(urlid,wordid,location)')  # 创建url-word-location数据表（每个链接下出现的所有单词和单词出现的位置）
        self.curs.execute('create table link(fromid integer,toid integer)')  # 创建url-url数据表（当前链接和目标链接对）
        self.curs.execute('create table linkwords(wordid,linkid)')  # 创建word-url数据表（链接描述）
        self.curs.execute('create index wordidx on wordlist(word)')  # 创建索引
        self.curs.execute('create index urlidx on urllist(url)')  # 创建索引
        self.curs.execute('create index wordurlidx on wordlocation(wordid)')  # 创建索引
        self.curs.execute('create index urltoidx on link(toid)')  # 创建索引
        self.curs.execute('create index urlfromidx on link(fromid)')  # 创建索引
        self.dbcommit()

    def getword(self,soup):
        # 获取每个单词
        text=self.gettextonly(soup)   #提取所有显示出来的文本
        allword=self.separatewords(text)  #使用分词器进行分词
        return allword

    # 根据一个网页源代码提取文字（不带标签的）。由外至内获取文本元素。style和script内不要
    def gettextonly(self,soup):
        v=soup.string
        if v==None:
            c=soup.contents   # 直接子节点的列表，将<tag>所有儿子节点存入列表
            resulttext=''
            for t in c:
                if t.name=='style' or t.name=='script':   #当元素为style和script和None时不获取内容
                    continue
                subtext=self.gettextonly(t)
                resulttext+=subtext+'\n'
            return resulttext.strip()
        else:
            if isinstance(v,bs4.element.Comment):   #代码中的注释不获取
                return ''
            return v.strip()

    # 使用结巴分词，去除标点符号
    def separatewords(self,text):
        seg_list = jieba.cut(text, cut_all=False)  #使用结巴进行中文分词
        allword = []
        for word in seg_list:
            if word not in ignorewords:
                allword.append(word)
                # print(allword)
        return allword

    #爬虫主函数
    def crawl(self,url,host):
        if url in self.urls and self.urls[url]['hascrawl']:return  # 如果网址已经存在于爬取列表中，并且已经爬取过，则直接返回
        try:
            if url in self.urls:
                if self.urls[url]['hascrawl']:return
                else: self.urls[url]['hascrawl']=True
            else:
                self.urls[url]={}
                self.urls[url]['hascrawl']=True
            response=urllib.request.urlopen(url)
            text = str(response.read(), encoding='utf-8')
            soup = BeautifulSoup(text, 'html.parser')

            links = soup('a')
            for link in links:
                if ('href' in dict(link.attrs)):
                    newurl = urllib.parse.urljoin(url, link['href'])
                    if not host in newurl: continue  # 非服务范围网址不爬取，不记录
                    if newurl == url: continue  # 如果网址是当前网址，不爬取，不记录
                    if newurl.find("'") != -1: continue  # 包含'的链接不爬取，不记录
                    newurl = newurl.split('#')[0]  # 去掉位置部分
                    if newurl[0:4] == 'http':  # 只处理http协议
                        if newurl not in self.urls:  # 将链接加入爬取列表
                            self.urls[newurl] = {}
                            self.urls[newurl]['hascrawl'] = False
                        # 添加链接描述
                        linkText = self.gettextonly(link).strip()  # 获取链接的描述
                        self.addlinkref(url, newurl, linkText)  # 添加链接跳转对和链接描述

            self.addtoindex(url, soup.body)  # 创建连接索引和链接-分词库
            self.dbcommit()    #保存
            return True
        except:
            traceback.print_exc()
            return False
            # print("Could not parse page %s" % url)


    # 建立链接索引和链接-分词索引
    def addtoindex(self, url, soup):
        # 获得urlid
        urlid = self.get_add_id('urllist', 'url', url)
        # 获取每个单词
        allword = self.getword(soup)  # 提取所有显示出来的文本
        print(allword)
        # 将每个单词与该url关联，写入到数据库
        index = 0
        for word in allword:
            index += 1
            wordid = self.get_add_id('wordlist', 'word', word)
            self.curs.execute("insert into wordlocation(urlid,wordid,location) values (%d,%d,%d)" % (urlid, wordid, index))

    # 添加链接跳转对，和链接-描述文本。
    def addlinkref(self, urlFrom, urlTo, linkText):
        words = self.separatewords(linkText)
        fromid = self.get_add_id('urllist', 'url', urlFrom)   #参数：表名、列名、值
        toid = self.get_add_id('urllist', 'url', urlTo)   #参数：表名、列名、值
        if fromid == toid: return
        cur = self.curs.execute("insert into link(fromid,toid) values (%d,%d)" % (fromid, toid))
        linkid = cur.lastrowid
        for word in words:
            wordid = self.get_add_id('wordlist', 'word', word)   #参数：表名、列名、值
            self.curs.execute("insert into linkwords(linkid,wordid) values (%d,%d)" % (linkid, wordid))

    # 辅助函数，用于获取数据库中记录的id，并且如果记录不存在，就将其加入数据库中，再返回id
    def get_add_id(self, table, field, value):
        command = "select rowid from %s where %s='%s'" % (table, field, value)
        cur = self.curs.execute(command)
        res = cur.fetchall()
        if res == None or len(res) == 0:
            cur = self.curs.execute("insert into %s (%s) values ('%s')" % (table, field, value))
            return cur.lastrowid
        else:
            return res[0][0]   #返回第一行第一列

    # （每个链接的pagerank=指向此链接的网页的pagerank/网页中的链接总数*0.85+0.15，其中0.85表示阻尼因子，表示网页是否点击该网页中的链接）
    # pagerank算法，离线迭代计算，形成每个链接的稳定pagerank值。
    def calculatepagerank(self, iterations=20):
        # 清除您当前的pagerank表
        self.curs.execute('drop table if exists pagerank')
        self.curs.execute('create table pagerank(urlid primary key,score)')

        # 初始化每个url，令其pagerank的值为1
        for (urlid,) in self.curs.execute('select rowid from urllist').fetchall():
            self.curs.execute('insert into pagerank(urlid,score) values (%d,1.0)' % urlid)
        self.dbcommit()

        for i in range(iterations):
            print("Iteration %d" % (i))
            for (urlid,) in self.curs.execute('select rowid from urllist').fetchall():
                pr = 0.15
                # 循环遍历指向当前网页的所有其他网页
                for (linker,) in self.curs.execute('select distinct fromid from link where toid=%d' % urlid).fetchall():
                    # 得到链接源对应网页的pagerank值
                    linkingpr = self.curs.execute('select score from pagerank where urlid=%d' % linker).fetchall()[0][0]
                    # 根据链接源求总的链接数
                    linkingcount = self.curs.execute('select count(*) from link where fromid=%d' % linker).fetchall()[0][0]
                    pr += 0.85 * (linkingpr / linkingcount)
                self.curs.execute('update pagerank set score=%f where urlid=%d' % (pr, urlid))
            self.dbcommit()


# 爬取指定域名范围内的所有网页，beginurl为开始网址，host为根网址
def crawlerhost(beginurl,host,dbname):
    mycrawler = crawler(dbname)      #定义爬虫对象
    mycrawler.crawl(beginurl,host)  #爬取主页
    for url in list(mycrawler.urls.keys()):
        print(url)
        mycrawler.crawl(url, host)  # 爬取子网页
    # 获取pageRank数据
    mycrawler.calculatepagerank()


# 读取数据库信息，检验是否成功建立了搜索数据库
def readdb(dbname):
    con = sqlite3.connect(dbname)
    curs = con.cursor()
    command = "select fromid,toid from link"   #'select * from link'
    cur = curs.execute(command)
    res = cur.fetchall()
    allurl=[]
    if res != None and len(res) > 0:
        for row in res:
            print(row)
            command = "select url from urllist where rowid=%d" % row[0]
            fromurl = curs.execute(command).fetchall()[0][0]
            command = "select url from urllist where rowid=%d" % row[1]
            tourl = curs.execute(command).fetchall()[0][0]
            # print(fromurl,tourl)
            if fromurl not in allurl:
                allurl.append(fromurl)
            if tourl not in allurl:
                allurl.append(tourl)





if __name__ == '__main__':
    # 爬取服务器建立数据库
    url = 'http://blog.csdn.net/luanpeng825485697'
    # if os._exists('csdn.db'):
    #     os.remove('csdn.db')    #删除旧的数据库
    crawlerhost(url, url,'csdn.db')

    #读取数据库
    readdb('csdn.db')




