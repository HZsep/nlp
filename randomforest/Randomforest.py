import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split , GridSearchCV, KFold
import pandas as pd
import  joblib
RF = RandomForestClassifier(random_state=42,max_depth= 9 , min_samples_leaf= 2, min_samples_split=5,n_estimators=120)
DT = DecisionTreeClassifier(max_depth=5 )
data =  pd.read_csv('/Users/zfy/Documents/titanic/train.csv')
data.loc[: , 'Sex'] = (data['Sex'] == 'male').astype('int')
data['Embarked']=data['Embarked'].fillna('C')
data['Age'] = data['Age'].fillna(data['Age'].mean())
dummy = pd.get_dummies(data['Embarked'],prefix='Embarked')
print(dummy)
data = pd.concat([data , dummy],axis=1)
print(data)
data['Fare']=data[['Fare']].fillna(data.groupby('Pclass').transform(np.mean))
data['Group_ticket']=data['Fare'].groupby(data['Ticket']).transform('count')
data['Fare']=data['Fare']/data['Group_ticket']
print(data)
data['Fare_bins']=pd.qcut(data['Fare'],5)
data['Fare_bin_id'] = pd.factorize(data['Fare_bins'])[0]
fare_bin_dummies = pd.get_dummies(data['Fare_bin_id']).rename(columns=lambda x: 'Fare_' + str(x))
data = pd.concat([data, fare_bin_dummies], axis=1)
data.drop(['Fare_bins'], axis=1, inplace=True)
def family_size_category(family_size):
    if family_size <= 0:
        return ('Single')
    elif family_size <= 3:
        return ('Small_Family')
    else:
        return ('Large_Family')
data['Family_Size'] = data['Parch'] + data['SibSp']
data['Family_Size_Category'] = data['Family_Size'].map(family_size_category)
family_size_dummy = pd.get_dummies(data['Family_Size_Category'])
data = pd.concat([data, family_size_dummy], axis=1)
data['Family_Size_Category'] = pd.factorize(data['Family_Size_Category'])[0]
data.loc[data['Cabin'].isnull(),'Cabin']='N'
data.drop(['Cabin'],axis = 1,inplace=True)
print(data)
data.drop(['PassengerId', 'Embarked', 'Name',  'Fare_bin_id', 'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'],axis=1,inplace = True)
#print(data)
#x = data.iloc[:, data.columns != 'Survived']
#print(x)
#y = data.iloc[:, data.columns == 'Survived']
'''
Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y ,test_size=0.3,random_state=42)
param2 = {'n_estimators': list(range(50,200,10)), 'max_depth': list(range(3,10,1)),'min_samples_split':list(range(5,10,1))
    ,'criterion': ['gini', 'entropy','mse']
    ,'min_samples_leaf':list(range(1,10,1))}
gsearch_2 = GridSearchCV(RF, param_grid=param2, cv=3, scoring='accuracy',)
rdmf = gsearch_2.fit(Xtrain , Ytrain)
print(rdmf.best_params_)
print('best accuracy_test : %f'% rdmf.best_score_)
'''
# 十折交叉
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(data):
    print("Train:", train_index, "Validation:",test_index)
    print(len(train_index),len(test_index))
X_train, X_test = data.iloc[train_index, data.columns != 'Survived'], data.iloc[test_index, data.columns != 'Survived']
y_train ,y_test = data.iloc[train_index, data.columns == 'Survived'], data.iloc[test_index, data.columns == 'Survived']
print('X_train :',X_train)
print(len(X_train),len(X_test))
rdmf = RF.fit(X_train , y_train)
score = rdmf.score(X_test,y_test)
print(score)
joblib.dump(rdmf,'train_model.m')