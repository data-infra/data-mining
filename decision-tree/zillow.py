# 对住房价格进行估计。api网站不能用了
import xml.dom.minidom
import urllib.request
import treepredict

zwskey="X1-ZWzlchwxis15aj_9skq6"

def getaddressdata(address,city):
    escad=address.replace(' ','+')
    # 构造url
    url='http://www.zillow.com/webservice/GetDeepSearchResults.htm?'
    url+='zws-id=%s&address=%s&citystatezip=%s' % (zwskey,escad,city)
    # 解析xml形式的返回结果
    request = urllib.request.Request(url)
    doc=xml.dom.minidom.parseString(urllib.request.urlopen(request).read())
    print(url)
    code=doc.getElementsByTagName('code')[0].firstChild.data
    # 状态码为0代表操作成功，否则代表错误发生
    if code!='0': return None
    # 提取有关房产的信息
    try:
        zipcode=doc.getElementsByTagName('zipcode')[0].firstChild.data
        use=doc.getElementsByTagName('useCode')[0].firstChild.data
        year=doc.getElementsByTagName('yearBuilt')[0].firstChild.data
        sqft=doc.getElementsByTagName('finishedSqFt')[0].firstChild.data
        bath=doc.getElementsByTagName('bathrooms')[0].firstChild.data
        bed=doc.getElementsByTagName('bedrooms')[0].firstChild.data
        rooms=1         #doc.getElementsByTagName('totalRooms')[0].firstChild.data
        price=doc.getElementsByTagName('amount')[0].firstChild.data
    except:
        return None

    return (zipcode,use,int(year),float(bath),int(bed),int(rooms),price)

# 读取文件构造数据集
def getpricelist():
    l1=[]
    for line in open('addresslist.txt'):
        data=getaddressdata(line.strip(),'Cambridge,MA')
        print(data)
        l1.append(data)
    return l1



if __name__=='__main__':  #只有在执行当前模块时才会运行此函数
    housedata = getpricelist()
    print(housedata)
    tree = treepredict.buildtree(housedata,scoref=treepredict.variance)   #创建决策树
    treepredict.drawtree(tree,'house.jpg')