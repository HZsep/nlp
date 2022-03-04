#观察数据
import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np
import seaborn as sns
data = pd.read_csv('/Users/zfy/Downloads/house-prices-advanced-regression-techniques/train.csv')
'''
var = 'MSSubClass'
data = pd.concat([data['SalePrice'],data[var]],axis = 1)
data.plot.scatter(x = var, y = 'SalePrice',ylim=(0, 800000) )
plt.show()
'''
#如图一不符合线性不保留
'''
var = 'MSZoning'
labels = data[var].unique().tolist()
print(labels)
data = pd.concat([data['SalePrice'],data[var]],axis = 1)
data.plot.scatter(x = var, y = 'SalePrice',ylim=(0, 800000))
plt.show()
'''
#如图二不符合线性不保留
'''
var  = 'GrLivArea'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()
'''
#如图三符合线性保留
''''
var  = 'LotFrontage'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()
'''
#基本符合，可以考虑，接下来处理数据
'''
var  = 'LotFrontage'
data[var] = data[var].fillna(data[var].mean())
data = pd.concat([data['SalePrice'], data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()
'''
#留下
'''
var  = 'LotArea'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()
'''
#舍弃
'''
var  = 'Street'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()
'''
#无意义，舍弃
'''
var = 'Alley'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
labels = data[var].unique().tolist()
data[var] = data[var].apply(lambda x:labels.index(x))
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()
'''
#同理
'''
var = 'LotShape'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
labels = data[var].unique().tolist()
data[var] = data[var].apply(lambda x:labels.index(x))
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()
'''
#同理
'''
var = 'LandContour'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
labels = data[var].unique().tolist()
data[var] = data[var].apply(lambda x:labels.index(x))
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()
'''
'''
var = 'Neighborhood'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
labels = data[var].unique().tolist()
data[var] = data[var].apply(lambda x:labels.index(x))
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()
'''
#同理
#可以看出接下来'Condition1',Condition2','BldgType','HouseStyle'全部不符
'''
var = 'OverallQual'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice",ylim=(0,800000))
plt.show()
'''
#图七，明显符合
'''
var = 'OverallCond'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice",ylim=(0,800000))
plt.show()
'''
#不符
'''
var = 'YearBuilt'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice",ylim=(0,800000))
plt.show()
'''
#图八，符合
'''
var = 'YearRemodAdd'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice",ylim=(0,800000))
plt.show()
'''
#图九，符合
#接下来不考虑任何非数值型
'''
var = 'TotRmsAbvGrd'
data = pd.concat([data['SalePrice'], data[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice",ylim=(0,800000))
plt.show()
'''
#符合
'''
.................
'''
#最后绘制关系图
'''
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.pairplot(data[cols], size = 2.5)
plt.show()
'''
#绘制热力图
corrmat = data.corr()
k  = 81
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=0.6)
hm = sns.heatmap(cm, cbar=True, annot=True, \
                 square=True, fmt='.2f', annot_kws={'size':3}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#选取
#OverallQual
#YearBuilt
#TotalBsmtSF
#1stFlrSF
#GrLivArea
#FullBath
#TotRmsAbvGrd
#GarageCars
#GarageArea