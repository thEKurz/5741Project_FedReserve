# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 02:53:10 2021

@author: Kelvin
"""

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

Text_DF=pd.read_csv('text_sent_1.csv',index_col='Date',parse_dates=['Date'])
Text_DF_1=pd.read_csv('text_sent_flair.csv',index_col='Date',parse_dates=['Date'])
Text_DF_2=pd.read_csv('text_sent_textblob.csv',index_col='Date',parse_dates=['Date'])
I_df=pd.read_csv('Independant_Numeric_Variables.csv',index_col='DATE',parse_dates=['DATE'])
D_df=pd.read_csv('Dependant_Numeric_Variables.csv',index_col='DATE',parse_dates=['DATE'])
D_df.fillna(method='ffill',inplace=True)

I_df[Text_DF.columns]=Text_DF[Text_DF.columns]
I_df[Text_DF_1.columns]=Text_DF_1[Text_DF_1.columns]
I_df[Text_DF_2.columns]=Text_DF_2[Text_DF_2.columns]
I_df.fillna(method='ffill',inplace=True)


# Fill NaN via matrix imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(I_df)
I_df_transform = imp.transform(I_df)

I_df_transform = pd.DataFrame(I_df_transform)
I_df_transform.columns = I_df.columns
I_df_transform.index = I_df.index

# Compute lagged variables
def TS_pipe(X,window_size=24):
    df=X
    df_1=df
    for window in range(1, window_size + 1):
        shifted = df_1.shift(window)
        df=df.join(shifted.rename(columns=lambda x: x+ "_" + str(window) + "_lag"))
    for x in df_1.columns:
        df[str(x) + '_MA_' + str(window_size)] = df[x].rolling(window=window_size).mean()
    return df

def variable_combos(X):
    new_df=X.copy()
    col_pairs = list(itertools.combinations(new_df.columns, 2))
    for a,b in col_pairs:
        new_df[str(a)+ "," + str(b)]=new_df[a]*new_df[b]
    return new_df

I_df24 = TS_pipe(I_df_transform, window_size = 24)
I_df24=I_df24[~(I_df24.index<'2003-01-01')]
D_df=D_df[~(D_df.index<'2003-01-01')]

# Split data into train, validation and test
kf = sklearn.model_selection.KFold(n_splits=5)
kf.get_n_splits(I_df24)

# Test for best Lasso model
from sklearn.linear_model import Lasso

def lasso_test(X_train, X_test, y_train, y_test, al = 1):
    m1 = Lasso(alpha=al, fit_intercept=True, max_iter=1000)
    m1.fit(X_train, y_train)
    sc= m1.score(X_test, y_test)
    return sc
    
alpha_range = np.arange(0.01, 1.01, 0.01)
score_list = list()
for al in alpha_range:
    avg_score = 0
    for train_index, test_index in kf.split(I_df24):
        X_train, X_test = I_df24.iloc[train_index,:], I_df24.iloc[test_index,:]
        y_train, y_test = D_df.iloc[train_index,:], D_df.iloc[test_index,:]
        avg_score += lasso_test(X_train, X_test, y_train, y_test, al)
    score_list.append(avg_score/5)

plt.plot(alpha_range, score_list)

