# 利用分类器，应用到过滤博客订阅源
import feedparser
import re
import docclass

# 接受一个博客订阅源的url文件名并对内容项进行分类
def read(feedfile,classifier):
    # 得到订阅源的内容项并遍历循环
    f=feedparser.parse(feedfile)
    for entry in f['entries']:
        print
        print('-----')
        # 将内容项打印输出
        print('Title:     '+entry['title'])
        print('Publisher: '+entry['publisher'])
        print
        print(entry['summary'])


        # 将所有文本组合在一起，为分类器构建一个内容项
        fulltext='%s\n%s\n%s' % (entry['title'],entry['publisher'],entry['summary'])

        # 将当前分类的最佳推测结果打印输出
        print('Guess: '+str(classifier.classify(fulltext)))

        # 请求用户给出正确分类，并据此进行训练
        if(entry.has_key('cat') and entry['cat']!=None):
            classifier.train(fulltext, entry['cat'])
        else:
            cl=input('Enter category: ')
            classifier.train(fulltext,cl)


# 对特征检测的改进。新的特征提取函数。更关注文章的名称、摘要、作者介绍
def entryfeatures(entry):
    splitter=re.compile('\\W*')
    features={}

    # 提取标题中的单词并进行标识
    titlewords=[s.lower() for s in splitter.split(entry['title']) if len(s)>2 and len(s)<20]
    for w in titlewords: features['Title:'+w]=1

    # 提取摘要中单词
    summarywords=[s.lower() for s in splitter.split(entry['summary']) if len(s)>2 and len(s)<20]

    # 统计大写单词
    uc=0
    for i in range(len(summarywords)):
        w=summarywords[i]
        features[w]=1
        if w.isupper(): uc+=1

        # 将从摘要中获得词组作为特征
        if i<len(summarywords)-1:
            twowords=' '.join(summarywords[i:i+1])
            features[twowords]=1

    # 保持文章创建者和发布者名字的完整性
    features['Publisher:'+entry['publisher']]=1

    # UPPERCASE是一个“虚拟”单词，用以指示存在过多的大写内容
    if float(uc)/len(summarywords)>0.3: features['UPPERCASE']=1

    return features


if __name__=="__main__":     #只有在执行当前模块时才会运行此函数
    # 对博客文章进行分类和训练
    cl=docclass.fisherclassifier(docclass.getwords)
    cl.setdb('python_feed.db')
    read('python_search.xml',cl)

    # 使用改进的特征提取函数对文章分类进行处理
    cl = docclass.fisherclassifier(entryfeatures)
    cl.setdb('python_feed.db')
    read('python_search.xml', cl)

