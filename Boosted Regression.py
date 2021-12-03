# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:56:31 2021

@author: payma
"""

import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
from matplotlib import pyplot



I_df_MA_1=scaler.fit_transform(I_df_MA_post_2003)
I_df_MA_1 = pd.DataFrame(I_df_MA_1)
I_df_MA_1.columns = I_df_MA_1.columns
I_df_MA_1.index = I_df_MA_1.index

import xgboost as xgb

import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

kf = sklearn.model_selection.KFold(n_splits=5)
kf.get_n_splits(I_df_MA_1)

from sklearn.metrics import mean_squared_error as MSE


from sklearn.metrics import accuracy_score


accuracyscores = []


#scores for different learning rates
for i in range(1, 10):
    for d in range(D_df_post_2003.shape[1]):
        y1 = D_df_post_2003.iloc[:,[d,]]
        
        for train_index, test_index in kf.split(I_df_MA_1):
            X_train = np.asarray(I_df_MA_1.iloc[train_index,:])
            X_test = np.asarray(I_df_MA_1.iloc[test_index,:])
            y1_train = np.asarray(y1.iloc[train_index,:])
            y1_test = np.asarray(y1.iloc[test_index,:])
    
        
        xgb_r = xgb.XGBRegressor(n_estimators = 50, max_depth = 25, learning_rate = i/100,  objective='reg:squarederror')
        xgb_r.fit(X_train, y1_train)
        acc1 = xgb_r.score(X_test, y1_test)
        accuracyscores.append(acc1)
        

D_df.shape[1]

data = {'LR = .1':accuracyscores[0:17],
        'LR = .2':accuracyscores[17:34],
        'LR = .3':accuracyscores[34:51],
        'LR = .4':accuracyscores[51:68],
        'LR = .5':accuracyscores[68:85],
        'LR = .6':accuracyscores[85:102],
        'LR = .7':accuracyscores[102:119],
        'LR = .8':accuracyscores[119:136],
        'LR = .9':accuracyscores[136:]}
  
# Create DataFrame
df = pd.DataFrame(data)
df

#best model was with LR = .3
#Look at feature importance for this model

xgb_r = xgb.XGBRegressor(n_estimators = 10, max_depth = 5, learning_rate = .3,  objective='reg:squarederror')
xgb_r.fit(X_train, y1_train)

pyplot.bar(range(len(xgb_r.feature_importances_)), xgb_r.feature_importances_)
pyplot.title('Boosted Regression Model Feature Importance')
pyplot.xlabel('Independent Variable Features')
pyplot.ylabel('Feature Importance')
pyplot.show()

LR = []
for j in range(1, 10):
    LR.append(j/10)
    
print(LR)


for d in range(0,17):
    plt.plot(LR, df.iloc[d])
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Model Accuracy for Different Step Sizes")
plt.show()

plt.plot(LR, df.iloc[5], label='GS1')
plt.plot(LR, df.iloc[8], label='GS5')
plt.plot(LR, df.iloc[13], label='UNRATE')
plt.legend()
plt.title("Model Accuracy for Different Step Sizes")
plt.show()



