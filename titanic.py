import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import  train_test_split
import numpy as np
from sklearn.externals import joblib
data = pd.read_csv('/Users/zfy/Documents/titanic/data.csv')
data.head(10)
#print(data)
#筛选特征
data.drop(['Cabin','Name','Ticket'],inplace=True,axis=1)
print(data)
#填补
data['Age']
print(data['Age'])
data['Age'] = data['Age'].fillna(data['Age'].mean())
print(data.info())
data = data.dropna()#只要有缺失就删除
labels = data['Embarked'].unique().tolist()#unique看有几个取值
data['Embarked'] = data['Embarked'].apply(lambda x:labels.index(x))#将索引转换为数字
print(data)
data.loc[:,'Sex']  = (data['Sex']== 'male').astype('int')#数字用iloc,loc
print(data)
x = data.iloc[:, data.columns != 'Survived']
print(x)
y = data.iloc[:, data.columns == 'Survived']
print(y)
Xtrain , Xtest , Ytrain ,Ytest = train_test_split(x, y, test_size=0.3)
for i in [Xtrain , Xtest , Ytrain ,Ytest]:
    i.index = range(i.shape[0])
clf = DecisionTreeClassifier(random_state=25)
clf = clf.fit(Xtrain,Ytrain)
score_ = clf.score(Xtest,Ytest)
print(score_)
tr = []
te = []
for i in range(10):
    clf = DecisionTreeClassifier(random_state=25
                                 ,max_depth=i+1
                                 ,criterion="entropy"
                                )
    clf = clf.fit(Xtrain, Ytrain)
    score_tr = clf.score(Xtrain,Ytrain)
    score_te = cross_val_score(clf,x,y,cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)
print(max(te))
plt.plot(range(1,11),tr,color="red",label="train")
plt.plot(range(1,11),te,color="blue",label="test")
plt.xticks(range(1,11))
plt.legend()
plt.show()
#网格搜索
gini_thresholds = np.linspace(0,0.5,20)
parameters = {'splitter':('best','random')
              ,'criterion':("gini","entropy")
              ,"max_depth":[*range(1,10)]
              ,'min_samples_leaf':[*range(1,50,5)]
              ,'min_impurity_decrease':[*np.linspace(0,0.5,20)]
}
clf = DecisionTreeClassifier(random_state=25)
GS = GridSearchCV(clf, parameters, cv=10)
GS.fit(Xtrain,Ytrain)
print(GS.best_params_)
print(GS.best_score_)

