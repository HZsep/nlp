import joblib
import pandas as pd
data =  pd.read_csv('/Users/zfy/Documents/titanic/test.csv')
data.loc[: , 'Sex'] = (data['Sex'] == 'male').astype('int')
labels = data['Embarked'].unique().tolist()#unique看有几个取值
data['Embarked'] = data['Embarked'].apply(lambda x:labels.index(x))#将索引转换为数字
data['Age'] = data['Age'].fillna(data['Age'].mean())
par = ['Pclass', 'Sex', 'Age', 'SibSp', 'Embarked']
rfr = joblib.load('./train_Res.m')
data_valid_x = pd.concat([data[par]],axis = 1)
x = data_valid_x.values
y_te_pred = rfr.predict(x)
print(y_te_pred)
print(y_te_pred.shape)
print(x.shape)
prediction = pd.DataFrame(y_te_pred, columns=['Survived'])
result = pd.concat([ data['PassengerId'], prediction], axis=1)
result.columns
result.to_csv('./Predictions.csv', index=False)