import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import  preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
#引入数据
data = pd.read_csv('/Users/zfy/Downloads/house-prices-advanced-regression-techniques/train.csv')
cols = ['OverallQual','YearBuilt','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']
x = data[cols].values
y = data['SalePrice'].values
x_scaled = preprocessing.StandardScaler().fit_transform(x)#使满足正态分布
y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))#同上
X_train,X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
pred = model.predict(X_test)
def figure(title, *datalist):
    plt.figure(facecolor='gray', figsize=[16, 8])
    for v in datalist:
        plt.plot(v[0], '-', label=v[1], linewidth=2)
        plt.plot(v[0], 'o')
    plt.grid()
    plt.title(title, fontsize=20)
    plt.legend(fontsize=16)
    plt.show()


y_train_pred = model.predict(X_train)
print('y_train_pred : ', y_train_pred)
# 计算均分方差
train_MSE = [mean_squared_error(y_train, [np.mean(y_train)] * len(y_train)),
             mean_squared_error(y_train, y_train_pred)]

# 计算平均绝对误差
train_MAE = [mean_absolute_error(y_train, [np.mean(y_train)] * len(y_train)),
             mean_absolute_error(y_train, y_train_pred)]
figure(' MSE = %.4f' % (train_MSE[-1]), [train_MSE, 'MSE'])
figure(' MAE = %.4f' % (train_MAE[-1]), [train_MAE, 'MAE'])

# 绘制预测值与真实值图
figure('预测值与真实值图 模型的' + r'$R^2=%.4f$' % (r2_score(y_train_pred, y_train)), [pred, 'pred_value'],
       [y_test, 'true_value'])
#保存模型
joblib.dump(model ,'train_model.m')