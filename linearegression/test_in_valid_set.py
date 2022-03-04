import pandas as pd
import numpy as np
import joblib
data = pd.read_csv('/Users/zfy/Downloads/house-prices-advanced-regression-techniques/test.csv')
par = ['OverallQual','YearBuilt','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']
rfr = joblib.load('./train_model.m')
def ifNAN(par,data):
    for i in par:
        if(np.isnan(data[i]).any()):
             print("data "+ i + " has null" )
             print('proccessing.....')
             data[i] = data[i].fillna(data[i].mean())
        else:
              print('all data is complete')
ifNAN(par,data)
#验证一下是否填补
#labels = data['TotalBsmtSF'].unique().tolist()
#print(labels)
data_valid_x = pd.concat([data[par]],axis = 1)
x = data_valid_x.values
y_te_pred = rfr.predict(x)
print(y_te_pred)
print(y_te_pred.shape)
print(x.shape)
prediction = pd.DataFrame(y_te_pred, columns=['SalePrice'])
result = pd.concat([ data['Id'], prediction], axis=1)
result.columns
result.to_csv('./Predictions.csv', index=False)