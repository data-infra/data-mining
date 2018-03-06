# 对“热度”评价进行建模。api服务网站不能用了
import urllib.request
import treepredict
import xml.dom.minidom

api_key='479NUNJHETN'

def getrandomratings(c):
    # 为getRandomProfile构造url

    url="http://services.hotornot.com/rest/?app_key=%s" % api_key
    url+="&method=Rate.getRandomProfile&retrieve_num=%d" % c
    url+="&get_rate_info=true&meet_users_only=true"

    print(url)
    f1=urllib.request.urlopen(url).read()


    doc=xml.dom.minidom.parseString(f1)

    emids=doc.getElementsByTagName('emid')
    ratings=doc.getElementsByTagName('rating')  #获取评价

    # 将emids和ratings组合在一个列表中
    result=[]
    for e,r in zip(emids,ratings):
        if r.firstChild!=None:
            result.append((e.firstChild.data,r.firstChild.data))
    print(result)
    return result

stateregions={'New England':['ct','mn','ma','nh','ri','vt'],
              'Mid Atlantic':['de','md','nj','ny','pa'],
              'South':['al','ak','fl','ga','ky','la','ms','mo',
                       'nc','sc','tn','va','wv'],
              'Midwest':['il','in','ia','ks','mi','ne','nd','oh','sd','wi'],
              'West':['ak','ca','co','hi','id','mt','nv','or','ut','wa','wy']}

def getpeopledata(ratings):
    result=[]
    for emid,rating in ratings:
        # 对应于MeetMe.getProfile方法调用的url
        url="http://services.hotornot.com/rest/?app_key=%s" % api_key
        url+="&method=MeetMe.getProfile&emid=%s&get_keywords=true" % emid

        # 得到所有关于此人的详细信息
        try:
            rating=int(float(rating)+0.5)
            doc2=xml.dom.minidom.parseString(urllib.request.urlopen(url).read())
            gender=doc2.getElementsByTagName('gender')[0].firstChild.data
            age=doc2.getElementsByTagName('age')[0].firstChild.data
            loc=doc2.getElementsByTagName('location')[0].firstChild.data[0:2]

            # 将州转换为地区
            for r,s in stateregions.items():
                if loc in s: region=r

            if region!=None:
                result.append((gender,int(age),region,rating))
        except:
            pass
    return result

if __name__=='__main__':  #只有在执行当前模块时才会运行此函数
    l1=getrandomratings(500)
    print(len(l1))
    pdata = getpeopledata(l1)
    print(pdata)
    tree = treepredict.buildtree(pdata,scoref=treepredict.variance)   #创建决策树
    treepredict.prune(tree,0.5)   #剪支
    treepredict.drawtree(tree,'hot.jpg')