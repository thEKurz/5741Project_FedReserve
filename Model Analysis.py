# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 12:18:24 2021

@author: evank
"""

import pandas as pd
import sklearn as sk

#bring in dataframes
Text_DF=pd.read_csv('text_sent_1.csv',index_col='Date',parse_dates=['Date'])
I_df=pd.read_csv('Independant_Numeric_Variables.csv',index_col='DATE',parse_dates=['DATE'])
D_df=pd.read_csv('Dependant_Numeric_Variables.csv',index_col='DATE',parse_dates=['DATE'])


#Text_DF.set_index(pd.to_datetime(Text_DF.index), inplace=True)

I_df[Text_DF.columns]=Text_DF[Text_DF.columns]
I_df.fillna(method='ffill',inplace=True)
I_df.fillna(0,inplace=True)

def TS_pipe(X,window_size=24):
    df=(X/(X.max()-X.min()))
    for x in df.columns:
        df[str(x) + '_MA_' + str(window_size)] = df[x].rolling(window=window_size).mean()
    df_1=df
    for window in range(1, window_size + 1):
        shifted = df_1.shift(window)
        df=df.join(shifted.rename(columns=lambda x: x+ "_" + str(window) + "_lag"))
    return df

I_df_norm=TS_pipe(I_df)

I_df_post_2003=I_df_norm[~(I_df_norm.index<'2003-01-01')]
D_df_post_2003=D_df[~(D_df.index<'2003-01-01')]

D_df_norm= (D_df_post_2003/(D_df_post_2003.max()-D_df_post_2003.min()))

def model_analysis(X,y):
    X_train, X_test, y_train, y_test = rain_test_split(X, y, test_size=0.25, random_state=42)
    