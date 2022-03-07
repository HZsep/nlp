from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
LR = LogisticRegression()
enc = preprocessing.OneHotEncoder()
data =  pd.read_csv('/Users/zfy/Documents/titanic/train.csv')
data.loc[: , 'Sex'] = (data['Sex'] == 'male').astype('int')
labels = data['Embarked'].unique().tolist()#unique看有几个取值
data['Embarked'] = data['Embarked'].apply(lambda x:labels.index(x))#将索引转换为数字
data['Age'] = data['Age'].fillna(data['Age'].mean())
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Embarked','Survived']
data = data[cols]
print(data)
x = data.iloc[: , data.columns != 'Survived']
y = data.iloc[: , data.columns == 'Survived']
Xtrain , Xtest , Ytrain , Ytest = train_test_split(x ,y ,test_size=0.33,random_state=46)
Res = LR.fit(Xtrain,Ytrain)
Score = Res.score(Xtest ,Ytest)
print(Score)
