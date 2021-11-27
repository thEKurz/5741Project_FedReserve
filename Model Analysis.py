# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 12:18:24 2021

@author: evank
"""

import pandas as pd
from oboe import AutoLearner, error  # This may take around 15 seconds at first run.
import time
import numpy as np
import itertools

#bring in dataframes
Text_DF=pd.read_csv('text_sent_1.csv',index_col='Date',parse_dates=['Date'])
Text_DF_1=pd.read_csv('text_sent_flair.csv',index_col='Date',parse_dates=['Date'])
Text_DF_2=pd.read_csv('text_sent_textblob.csv',index_col='Date',parse_dates=['Date'])
I_df=pd.read_csv('Independant_Numeric_Variables.csv',index_col='DATE',parse_dates=['DATE'])
D_df=pd.read_csv('Dependant_Numeric_Variables.csv',index_col='DATE',parse_dates=['DATE'])
D_df.fillna(method='ffill',inplace=True)

#Text_DF.set_index(pd.to_datetime(Text_DF.index), inplace=True)

I_df[Text_DF.columns]=Text_DF[Text_DF.columns]
I_df[Text_DF_1.columns]=Text_DF_1[Text_DF_1.columns]
I_df[Text_DF_2.columns]=Text_DF_2[Text_DF_2.columns]
I_df.fillna(method='ffill',inplace=True)
I_df.fillna(0,inplace=True)

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
    for p in X.columns:
        new_df[str(p)+ "^2"]= X[p]**2
    col_pairs = list(itertools.combinations(new_df.columns, 2))
    for a,b in col_pairs:
        new_df[str(a)+ "," + str(b)]=new_df[a]*new_df[b]
    return new_df

I_df_post_2003=I_df[~(I_df.index<'2003-01-01')]
D_df_post_2003=D_df[~(D_df.index<'2003-01-01')]




from sklearn.model_selection import train_test_split


#linear regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

I_df_linear_reg=TS_pipe(variable_combos(I_df))
I_df_LR_post_2003=I_df_linear_reg[~(I_df_linear_reg.index<'2003-01-01')]

lasso_coef={}
lasso_alpha={}
for y in D_df_post_2003.columns:
    X=scaler.fit_transform(I_df_LR_post_2003)
    Y=scaler.fit_transform(D_df_post_2003[y])
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(Y), test_size=0.25, random_state=42)
    reg = LassoCV(cv=5, random_state=0).fit(X_train, y_train)
    lasso_coef[y]=reg.coef_
    lasso_alpha[y]=reg.alpha_   
    
    
def model_analysis(X,y):

    
    """
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(np.array(X), np.array(y), test_size=0.25, random_state=42)
    m = AutoLearner(p_type='regression', runtime_limit=30, method='Oboe',ensemble=True, verbose=False)
    # fit autolearner on training set and record runtime
    start = time.time()
    m.fit(X_train, y_train)
    elapsed_time = time.time() - start
    
    # use the fitted autolearner for prediction on test set
    y_predicted = m.predict(X_test)
    print("prediction error: {}".format(error(np.array(y_test), y_predicted, 'classification')))    
    print("elapsed time: {}".format(elapsed_time))
    m.get_models()
    """
    
    
    
    return
