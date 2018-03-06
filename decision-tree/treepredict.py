# 使用决策树对web站点的用户行为及最终购买行为（选择的服务类型）进行预测。
# 决策树同过找出最重要的分类属性，再找出第二重要的分类属性。一次向下，建立树的层次。
# 属性的重要程度是根据使用该属性进行分类后，两个类别的基尼不纯度（熵）的大小判定的

# 下面的数据分为为每个用户的来源网站、位置、是否阅读FAQ、浏览网页数目、选择的服务类型（目标结果）
my_data=[['slashdot','USA','yes',18,'None'],
         ['google','France','yes',23,'Premium'],
         ['digg','USA','yes',24,'Basic'],
         ['kiwitobes','France','yes',23,'Basic'],
         ['google','UK','no',21,'Premium'],
         ['(direct)','New Zealand','no',12,'None'],
         ['(direct)','UK','no',21,'Basic'],
         ['google','USA','no',24,'Premium'],
         ['slashdot','France','yes',19,'None'],
         ['digg','USA','no',18,'None'],
         ['google','UK','no',18,'None'],
         ['kiwitobes','UK','no',19,'None'],
         ['digg','New Zealand','yes',12,'Basic'],
         ['slashdot','UK','no',21,'None'],
         ['google','UK','yes',18,'Basic'],
         ['kiwitobes','France','yes',19,'Basic']]


# 建立决策树上的节点类
class decisionnode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col                #待检测条件所属的列索引。即当前是对第几列数据进行分类
        self.value=value            #为使结果为true，当前列必须匹配的值
        self.results=results        #如果当前节点时叶节点，表示该节点的结果值，如果不是叶节点，为None
        self.tb=tb                  #判断条件为true后的子节点
        self.fb=fb                  #判断调节为false后的子节点

# 在某一列上对数据集合进行拆分，能够处理数值型数据或名词性数据。其实决策树只能处理离散型数据，对于连续性数据也是划分为范围区间块
# rows样本数据集，要匹配的列索引，指定列上的数据要匹配的值
def divideset(rows,column,value):
    # 定义一个函数，令其告诉我们数据行属于第一组（返回值为true）还是第二组（返回值false）
    split_function=None
    if isinstance(value,int) or isinstance(value,float):
        split_function=lambda row:row[column]>=value   #按大于和小于区分
    else:
        split_function=lambda row:row[column]==value   #按等于和不等于区分

    # 将数据集拆分成两个集合，并返回
    set1=[row for row in rows if split_function(row)]
    set2=[row for row in rows if not split_function(row)]
    return (set1,set2)


# 对各种可能的目标结果（选择的服务类型）进行计数（样本数据每一行数据的最后一列记录了目标结果）。rows样本数据
def uniquecounts(rows):
    results={}
    for row in rows:
        # 目标结果在样本数据最后一列
        r=row[len(row)-1]
        if r not in results: results[r]=0
        results[r]+=1
    return results

# 计算基尼不纯度，作为属性对分类的重要程度。（要计算根据属性划分的两个子集各自的基尼不纯度，不纯度越低，证明在各自的分类中越纯，则属性划分的越好）
# 计算随机放置的数据项出现于错误分类中的概率。rows样本数据集
def giniimpurity(rows):
    total=len(rows)
    counts=uniquecounts(rows)
    imp=0
    for k1 in counts:
        p1=float(counts[k1])/total
        for k2 in counts:
            if k1==k2: continue
            p2=float(counts[k2])/total
            imp+=p1*p2
    return imp

# 计算熵，作为属性对分类的重要程度。（要计算根据属性划分的两个子集各自的熵值，熵值越低，证明在各自的分类中越纯，则属性划分的越好）
# 熵是遍历所有可能的结果之后得到的p(x)log(p(x))之和。rows样本数据集
def entropy(rows):
    from math import log
    log2=lambda x:log(x)/log(2)
    results=uniquecounts(rows)
    # 此处开始计算熵的值
    ent=0.0
    for r in results.keys():
        p=float(results[r])/len(rows)
        ent=ent-p*log2(p)
    return ent

# 构建决策树.scoref为信息增益的计算函数
def buildtree(rows,scoref=entropy):
    if len(rows)==0: return decisionnode()
    current_score=scoref(rows)

    # 定义一些变量以记录最佳拆分条件
    best_gain=0.0
    best_criteria=None
    best_sets=None

    column_count=len(rows[0])-1
    for col in range(0,column_count):    #遍历每一列（除最后一列，因为最后一列是目标结果）
        # 在当前列中生成一个由不同值构成的序列
        column_values={}
        for row in rows:
            column_values[row[col]]=1
        # 接下来根据这一列中的每个值，尝试对数据集进行拆分
        for value in column_values.keys():
            (set1,set2)=divideset(rows,col,value)

            # 计算信息增益
            p=float(len(set1))/len(rows)
            gain=current_score-p*scoref(set1)-(1-p)*scoref(set2)
            if gain>best_gain and len(set1)>0 and len(set2)>0:   #找到信息增益最大的分类属性
                best_gain=gain
                best_criteria=(col,value)
                best_sets=(set1,set2)
    # 创建子分支
    if best_gain>0:
        trueBranch=buildtree(best_sets[0])   #创建分支
        falseBranch=buildtree(best_sets[1])  #创建分支
        return decisionnode(col=best_criteria[0],value=best_criteria[1],tb=trueBranch,fb=falseBranch)  #返回决策树节点
    else:
        return decisionnode(results=uniquecounts(rows))



#决策树的显示
def printtree(tree,indent=''):
    # 这是一个叶节点么？
    if tree.results!=None:
        print(str(tree.results))
    else:
        # 打印判断条件
        print(str(tree.col)+':'+str(tree.value)+'? ')

        # 打印分支
        print(indent+'T->',end='')
        printtree(tree.tb,indent+'  ')
        print(indent+'F->',end=''),
        printtree(tree.fb,indent+'  ')


from PIL import Image, ImageDraw

# 获取树的显示宽度
def getwidth(tree):
    if tree.tb==None and tree.fb==None: return 1
    return getwidth(tree.tb)+getwidth(tree.fb)

# 获取树的显示深度（高度）
def getdepth(tree):
    if tree.tb==None and tree.fb==None: return 0
    return max(getdepth(tree.tb),getdepth(tree.fb))+1

# 绘制树形图
def drawtree(tree,jpeg='tree.jpg'):
    w=getwidth(tree)*100
    h=getdepth(tree)*100+120

    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)

    drawnode(draw,tree,w/2,20)  #根节点坐标
    img.save(jpeg,'JPEG')

# 迭代画树的节点
def drawnode(draw,tree,x,y):
    if tree.results==None:
        # 得到每个分支的宽度
        w1=getwidth(tree.fb)*100
        w2=getwidth(tree.tb)*100

        # 确定此节点所要占据的总空间
        left=x-(w1+w2)/2
        right=x+(w1+w2)/2

        # 绘制判断条件字符串
        draw.text((x-20,y-10),str(tree.col)+':'+str(tree.value),(0,0,0))

        # 绘制到分支的连线
        draw.line((x,y,left+w1/2,y+100),fill=(255,0,0))
        draw.line((x,y,right-w2/2,y+100),fill=(255,0,0))

        # 绘制分支的节点
        drawnode(draw,tree.fb,left+w1/2,y+100)
        drawnode(draw,tree.tb,right-w2/2,y+100)
    else:
        txt=' \n'.join(['%s:%d'%v for v in tree.results.items()])
        draw.text((x-20,y),txt,(0,0,0))


# 对新的观测数据进行分类。observation为观测数据。tree为建立好的决策树
def classify(observation,tree):
    if tree.results!=None:
        return tree.results
    else:
        v=observation[tree.col]
        branch=None
        if isinstance(v,int) or isinstance(v,float):
            if v>=tree.value: branch=tree.tb
            else: branch=tree.fb
        else:
            if v==tree.value: branch=tree.tb
            else: branch=tree.fb
        return classify(observation,branch)


# 决策树剪枝。(因为有些属性的分类产生的熵值太小，没有区分的必要)，mingain为门限。
# 为了避免遇到大小台阶的问题（子树分支的属性比较重要），所以采取先建树，再剪支的方式
def prune(tree,mingain):
    # 如果分支不是叶节点，则对其进行剪枝操作
    if tree.tb.results==None:
        prune(tree.tb,mingain)
    if tree.fb.results==None:
        prune(tree.fb,mingain)

    # 如果两个自分支都是叶节点，则判断他们是否需要合并
    if tree.tb.results!=None and tree.fb.results!=None:
        # 构建合并后的数据集
        tb,fb=[],[]
        for v,c in tree.tb.results.items():
            tb+=[[v]]*c
        for v,c in tree.fb.results.items():
            fb+=[[v]]*c

        # 检查熵的减少情况
        delta=entropy(tb+fb)-(entropy(tb)+entropy(fb)/2)

        if delta<mingain:
            # 合并分支
            tree.tb,tree.fb=None,None
            tree.results=uniquecounts(tb+fb)

# 对包含缺失数据的新的观测数据进行分类。所有可能分支同时查询，最后计算加权结果
def mdclassify(observation,tree):
    if tree.results!=None:
        return tree.results
    else:
        v=observation[tree.col]
        if v==None:
            tr,fr=mdclassify(observation,tree.tb),mdclassify(observation,tree.fb)  #分别使用左右子树进行分类预测
            # 求统计结果
            tcount=sum(tr.values())
            fcount=sum(fr.values())
            tw=float(tcount)/(tcount+fcount)
            fw=float(fcount)/(tcount+fcount)
            result={}
            for k,v in tr.items(): result[k]=v*tw
            for k,v in fr.items(): result[k]=v*fw
            return result
        else:
            if isinstance(v,int) or isinstance(v,float):
                if v>=tree.value: branch=tree.tb
                else: branch=tree.fb
            else:
                if v==tree.value: branch=tree.tb
                else: branch=tree.fb
            return mdclassify(observation,branch)

# 如果输出结果不是分类而是数字，可以使用方差作为评价函数来取代熵或基尼不纯度
#计算数据集的统计方差
def variance(rows):
    if len(rows)==0: return 0
    data=[float(row[len(row)-1]) for row in rows]
    mean=sum(data)/len(data)
    variance=sum([(d-mean)**2 for d in data])/len(data)
    return variance






if __name__=='__main__':  #只有在执行当前模块时才会运行此函数
    tree = buildtree(my_data)   #创建决策树
    printtree(tree)
    drawtree(tree,jpeg='treeview.jpg')  #画树形图
    prune(tree,0.1)   #剪支
    result = mdclassify(['google',None,'yes',None],tree)   #使用决策树进行预测，新数据包含缺失数据
    print(result)

