# 获取百度新闻rss数据集，用于文本分类
import urllib
import re
from bs4 import BeautifulSoup
import json
import io

feedlist=[
    'http://news.baidu.com/n?cmd=1&class=civilnews&tn=rss&sub=0',  #国内焦点
    'http://news.baidu.com/n?cmd=1&class=shizheng&tn=rss&sub=0',   #时政焦点
    'http://news.baidu.com/n?cmd=1&class=gangaotai&tn=rss&sub=0',   #港澳台焦点
    'http://news.baidu.com/n?cmd=1&class=internews&tn=rss&sub=0',  #国际焦点
    'http://news.baidu.com/n?cmd=1&class=mil&tn=rss&sub=0',  #军事焦点
    'http://news.baidu.com/n?cmd=1&class=hqsy&tn=rss&sub=0',  #环球视野焦点
    'http://news.baidu.com/n?cmd=1&class=finannews&tn=rss&sub=0',  #财经焦点
    'http://news.baidu.com/n?cmd=1&class=stock&tn=rss&sub=0',  #股票焦点
    'http://news.baidu.com/n?cmd=1&class=money&tn=rss&sub=0',  #理财焦点
    'http://news.baidu.com/n?cmd=1&class=financialnews&tn=rss&sub=0',  #金融观察焦点
    'http://news.baidu.com/n?cmd=1&class=internet&tn=rss&sub=0',  #互联网焦点
    'http://news.baidu.com/n?cmd=1&class=rwdt&tn=rss&sub=0',  #人物动态焦点
    'http://news.baidu.com/n?cmd=1&class=gsdt&tn=rss&sub=0',  #公司动态焦点
    'http://news.baidu.com/n?cmd=1&class=housenews&tn=rss&sub=0',  #房产焦点
    'http://news.baidu.com/n?cmd=1&class=gddt&tn=rss&sub=0',  #各地动态焦点
    'http://news.baidu.com/n?cmd=1&class=zcfx&tn=rss&sub=0',  #政策风向焦点
    'http://news.baidu.com/n?cmd=1&class=fitment&tn=rss&sub=0',  #家居焦点
    'http://news.baidu.com/n?cmd=1&class=autonews&tn=rss&sub=0',  #汽车焦点
    'http://news.baidu.com/n?cmd=1&class=autobuy&tn=rss&sub=0',  #新车导购焦点
    'http://news.baidu.com/n?cmd=1&class=autoreview&tn=rss&sub=0',  #试驾焦点
    'http://news.baidu.com/n?cmd=1&class=sportnews&tn=rss&sub=0',  #体育焦点
    'http://news.baidu.com/n?cmd=1&class=nba&tn=rss&sub=0',  #NBA焦点
    'http://news.baidu.com/n?cmd=1&class=worldsoccer&tn=rss&sub=0',  #国际足球焦点
    'http://news.baidu.com/n?cmd=1&class=chinasoccer&tn=rss&sub=0',   #国内足球焦点
    'http://news.baidu.com/n?cmd=1&class=cba&tn=rss&sub=0',  #国内篮球焦点
    'http://news.baidu.com/n?cmd=1&class=othersports&tn=rss&sub=0',  #综合体育焦点
    'http://news.baidu.com/n?cmd=1&class=olympic&tn=rss&sub=0',  #奥运焦点
    'http://news.baidu.com/n?cmd=1&class=enternews&tn=rss&sub=0',  #娱乐焦点
    'http://news.baidu.com/n?cmd=1&class=star&tn=rss&sub=0',  #明星焦点
    'http://news.baidu.com/n?cmd=1&class=film&tn=rss&sub=0',  #电影焦点
    'http://news.baidu.com/n?cmd=1&class=tv&tn=rss&sub=0',  #电视焦点
    'http://news.baidu.com/n?cmd=1&class=music&tn=rss&sub=0',  #音乐焦点
    'http://news.baidu.com/n?cmd=1&class=gamenews&tn=rss&sub=0',  #游戏焦点
    'http://news.baidu.com/n?cmd=1&class=netgames&tn=rss&sub=0',  #网络游戏焦点
    'http://news.baidu.com/n?cmd=1&class=tvgames&tn=rss&sub=0',  #电视游戏焦点
    'http://news.baidu.com/n?cmd=1&class=edunews&tn=rss&sub=0',  #教育焦点
    'http://news.baidu.com/n?cmd=1&class=exams&tn=rss&sub=0',  #考试焦点
    'http://news.baidu.com/n?cmd=1&class=abroad&tn=rss&sub=0',  #留学焦点
    'http://news.baidu.com/n?cmd=1&class=healthnews&tn=rss&sub=0',  #健康焦点
    'http://news.baidu.com/n?cmd=1&class=baojian&tn=rss&sub=0',  #保健养生焦点
    'http://news.baidu.com/n?cmd=1&class=yiyao&tn=rss&sub=0',  #寻医问药焦点
    'http://news.baidu.com/n?cmd=1&class=technnews&tn=rss&sub=0',  #科技焦点
    'http://news.baidu.com/n?cmd=1&class=mobile&tn=rss&sub=0',  #手机焦点
    'http://news.baidu.com/n?cmd=1&class=digi&tn=rss&sub=0',  #数码焦点
    'http://news.baidu.com/n?cmd=1&class=computer&tn=rss&sub=0',  #电脑焦点
    'http://news.baidu.com/n?cmd=1&class=discovery&tn=rss&sub=0',  #科普焦点
    'http://news.baidu.com/n?cmd=1&class=socianews&tn=rss&sub=0',  #社会焦点
    'http://news.baidu.com/n?cmd=1&class=shyf&tn=rss&sub=0',  #社会与法焦点
    'http://news.baidu.com/n?cmd=1&class=shwx&tn=rss&sub=0',  #社会万象焦点
    'http://news.baidu.com/n?cmd=1&class=zqsk&tn=rss&sub=0',  #真情时刻焦点
]

def getrss1(feedlist):
    for url in feedlist:
        info={}
        info[url]={
            'title':'',
            'allitem':[]
        }
        try:
            response=urllib.request.urlopen(url)
            text = str(response.read(), encoding='utf-8')
            soup = BeautifulSoup(text, 'lxml')
            title = soup.title
            info[url]['title']=title
            for item in soup('item'):
                try:
                    print(item)
                    suburl={
                        'title':item('title').replace(']]>','').replace('<![CDATA[',''),
                        'link': item('link').replace(']]>', '').replace('<![CDATA[', ''),
                        'source': item('source').replace(']]>', '').replace('<![CDATA[', ''),
                        'text': item('description').get_text().replace(']]>',''),
                        'type':title
                    }
                    print(suburl)
                    info[url]['allitem'].append(suburl)
                except:
                    print('无法匹配'+item)
        except:
            print("error: %s" % url)


def getrss(feedlist):
    rss = {}

    for url in feedlist:
        rss[url] = {
            'title': '',
            'allitem': []
        }
        try:
            response = urllib.request.urlopen(url)
            text = str(response.read(), encoding='utf-8')
            soup = BeautifulSoup(text, 'lxml')
            title = soup.title.get_text()
            rss[url]['title'] = title
            patterstr = r'<item>.*?' \
                     r'<title>(.*?)</title>.*?' \
                     r'<link>(.*?)</link>.*?' \
                     r'<source>(.*?)</source>.*?' \
                     r'<description>.*?<br>(.*?)<br.*?' \
                     r'</item>'
            pattern = re.compile(patterstr,re.S)   #使用多行模式
            results = re.findall(pattern, text)   #如何查询多次

            if results!=None or len(results)==0:
                for result in results:
                    suburl = {
                        'title': result[0].replace(']]>', '').replace('<![CDATA[', ''),
                        'link': result[1].replace(']]>', '').replace('<![CDATA[', ''),
                        'source': result[2].replace(']]>', '').replace('<![CDATA[', ''),
                        'text': result[3].replace(']]>', ''),
                        'type': title
                    }
                    print(suburl)
                    rss[url]['allitem'].append(suburl)
        except:
            print("error: %s" % url)

    return rss


# 形成一个文本描述和分类的数据集。
if __name__ == '__main__':
    rss = getrss(feedlist)
    jsonstr = json.dumps(rss,ensure_ascii=False)
    f = io.open('rss.json', 'w', encoding='utf-8')
    f.writelines(jsonstr)
    f.close()








