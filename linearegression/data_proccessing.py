import pandas as pd
import  numpy as np
data = pd.read_csv('/Users/zfy/Downloads/house-prices-advanced-regression-techniques/train.csv')
#OverallQual

def ifNAN(par):
    for i in par:
         if(np.isnan(data[i]).any()):
             print('data'+ i +'has null' )
         else:
             print('all data is complete')
par = ['OverallQual','YearBuilt','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']
ifNAN(par)