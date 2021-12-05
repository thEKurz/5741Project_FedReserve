# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 12:18:24 2021

@author: evank
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

def MA(X,window_size=24):
    df=X.copy()
    for x in df.columns:
        df[str(x) + '_MA_' + str(window_size)] = df[x].rolling(window=window_size).mean()
    return df

def TS_pipe(X,window_size=24):
    df=X.copy()
    df_1=df
    for window in range(1, window_size + 1):
        shifted = df_1.shift(window)
        df=df.join(shifted.rename(columns=lambda x: x+ "_" + str(window) + "_lag"))
    for x in df_1.columns:
        df[str(x) + '_MA_' + str(window_size)] = df[x].rolling(window=window_size).mean()
    return df

def TS_lag_buckets(X,window_size=24,lag_buckets=4):
    bucket_size=int(window_size/lag_buckets)
    df=X.copy()
    df_1=df
    for window in np.arange(1, window_size+1,bucket_size):
        shifted = df_1.shift(window)
        for s in range(window,window+bucket_size):
            shifted += df_1.shift(s)
        shifted=shifted/bucket_size
        df=df.join(shifted.rename(columns=lambda x: x+ "_lag_" + str(window)+ "_"+str(window+bucket_size-1)))
    for x in df_1.columns:
        df[str(x) + '_MA_' + str(window_size)] = df[x].rolling(window=window_size).mean().copy()
    return df

def variable_combos(X):
    new_df=X.copy()
    col_pairs = list(itertools.combinations(new_df.columns, 2))
    for a,b in col_pairs:
        new_df[str(a)+ "," + str(b)]=new_df[a]*new_df[b]
    return new_df

#get a dataframe of coefficients from a coef attribute from your model
#and a dataframe of your original X variables
def get_coefs(coef_values,DF):
    coef_list=list(DF.columns)
    coef_df=pd.DataFrame(coef_values,index=coef_list)
    coef_df=coef_df.loc[~(coef_df==0).all(axis=1)]
    return coef_df



def coef_DF(coef_dict,i_df,R_2,R_val=.5):
    coef_df_list=[]
    for key in R_2:
        if R_2[key]>=R_val:
            coef=get_coefs(coef_dict[key],i_df)
            coef.columns=[key]
            coef_df_list.append(coef)
    coef_df = pd.concat(coef_df_list, axis=1)
    coef_df.sort_index()
    return coef_df

def acc_plot(y,y_pred,name,R2_score):
    plt.figure()
    plt.scatter(y,y_pred,color='g')
    plt.title('Predicted vs Actual value of ' + str(name)+ ' R^2 Score:'+str(R2_score))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    return plt


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

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(I_df)
I_df_transform = imp.transform(I_df)

I_df_transform = pd.DataFrame(I_df_transform)
I_df_transform.columns = I_df.columns
I_df_transform.index = I_df.index


I_df_post_2003=I_df_transform[~(I_df_transform.index<'2003-01-01')]
D_df_post_2003=D_df[~(D_df.index<'2003-01-01')]




from sklearn.model_selection import train_test_split


#linear regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

#lasso baseline
lasso_coef_BL={}
lasso_alpha_BL={}
lasso_model_BL={}
lasso_R2_BL={}
lasso_predict_BL={}
lasso_y_actual_BL={}
for y in D_df_post_2003.columns:
    print("Working on model for " + str(y))
    X=scaler.fit_transform(I_df_post_2003)
    Y=D_df_post_2003[y]
    n=y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y = LassoCV(cv=5, random_state=42,alphas=[.025,.05,.075,.1,.2,.3,.4,.5,.7],max_iter=5000).fit(X_train, y_train)
    lasso_coef_BL[n]=y.coef_
    lasso_alpha_BL[n]=y.alpha_   
    lasso_model_BL[n]=y
    lasso_R2_BL[n]=y.score(X_test,y_test)
    lasso_predict_BL[n]=y.predict(X_test)
    lasso_y_actual_BL[n]=y_test

coef_df_BL=coef_DF(lasso_coef_BL,I_df_post_2003,lasso_R2_BL,R_val=.5)


I_df_linear_reg=TS_pipe(variable_combos(I_df_transform))
I_df_LR_post_2003=I_df_linear_reg[~(I_df_linear_reg.index<'2003-01-01')]

lasso_coef={}
lasso_alpha={}
lasso_model={}
lasso_R2={}
lasso_predict={}
lasso_y_actual={}
for y in D_df_post_2003.columns:
    print("Working on model for " + str(y))
    X=scaler.fit_transform(I_df_LR_post_2003)
    Y=D_df_post_2003[y]
    n=y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y = LassoCV(cv=5, random_state=42,alphas=[.025,.05,.075,.1,.2,.3,.4,.5,.7],max_iter=5000).fit(X_train, y_train)
    lasso_coef[n]=y.coef_
    lasso_alpha[n]=y.alpha_   
    lasso_model[n]=y
    lasso_R2[n]=y.score(X_test,y_test)
    lasso_predict[n]=y.predict(X_test)
    lasso_y_actual[n]=y_test
    

#get coefficients from models with R^2 >.5
coef_df=coef_DF(lasso_coef,I_df_LR_post_2003,lasso_R2,R_val=.5)


    
#Lasso without combined features
I_df_TS=TS_pipe(I_df_transform)
I_df_TS_post_2003=I_df_TS[~(I_df_TS.index<'2003-01-01')]

lasso_coef_TS={}
lasso_alpha_TS={}
lasso_model_TS={}
lasso_R2_TS={}
lasso_predict_TS={}
lasso_y_actual_TS={}
for y in D_df_post_2003.columns:
    print("Working on model for " + str(y))
    X=scaler.fit_transform(I_df_TS_post_2003)
    Y=D_df_post_2003[y]
    n=y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y = LassoCV(cv=5, random_state=42,alphas=[.025,.05,.075,.1,.2,.3,.4,.5,.7],max_iter=5000).fit(X_train, y_train)
    lasso_coef_TS[n]=y.coef_
    lasso_alpha_TS[n]=y.alpha_   
    lasso_model_TS[n]=y
    lasso_R2_TS[n]=y.score(X_test,y_test)
    lasso_predict_TS[n]=y.predict(X_test)
    lasso_y_actual_TS[n]=y_test

#get coefficients from models with R^2 >.5
coef_df_TS=coef_DF(lasso_coef_TS,I_df_TS_post_2003,lasso_R2_TS,R_val=.5)


#Lasso with lag buckets and cobmined featured

I_df_lag_buckets=TS_lag_buckets(variable_combos(I_df_transform))
I_df_LB_post_2003=I_df_lag_buckets[~(I_df_lag_buckets.index<'2003-01-01')]

lasso_coef_LB={}
lasso_alpha_LB={}
lasso_model_LB={}
lasso_R2_LB={}
lasso_predict_LB={}
lasso_y_actual_LB={}
for y in D_df_post_2003.columns:
    print("Working on model for " + str(y))
    X=scaler.fit_transform(I_df_LB_post_2003)
    Y=D_df_post_2003[y]
    n=y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y = LassoCV(cv=5, random_state=42,alphas=[.025,.05,.075,.1,.2,.3,.4,.5,.7],max_iter=5000).fit(X_train, y_train)
    lasso_coef_LB[n]=y.coef_
    lasso_alpha_LB[n]=y.alpha_   
    lasso_model_LB[n]=y
    lasso_R2_LB[n]=y.score(X_test,y_test)
    lasso_predict_LB[n]=y.predict(X_test)
    lasso_y_actual_LB[n]=y_test

coef_df_LB=coef_DF(lasso_coef_LB,I_df_LB_post_2003,lasso_R2_LB,R_val=.5)


#Lasso with lag buckets only

I_df_LB=TS_lag_buckets(I_df_transform)
I_df_LB_post_2003_1=I_df_LB[~(I_df_LB.index<'2003-01-01')]

lasso_coef_LB_1={}
lasso_alpha_LB_1={}
lasso_model_LB_1={}
lasso_R2_LB_1={}
lasso_predict_LB_1={}
lasso_y_actual_LB_1={}
for y in D_df_post_2003.columns:
    print("Working on model for " + str(y))
    X=scaler.fit_transform(I_df_LB_post_2003_1)
    Y=D_df_post_2003[y]
    n=y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y = LassoCV(cv=5, random_state=42,alphas=[.025,.05,.075,.1,.2,.3,.4,.5,.7],max_iter=5000).fit(X_train, y_train)
    lasso_coef_LB_1[n]=y.coef_
    lasso_alpha_LB_1[n]=y.alpha_   
    lasso_model_LB_1[n]=y
    lasso_R2_LB_1[n]=y.score(X_test,y_test)
    lasso_predict_LB_1[n]=y.predict(X_test)
    lasso_y_actual_LB_1[n]=y_test

coef_df_LB_1=coef_DF(lasso_coef_LB_1,I_df_LB_post_2003_1,lasso_R2_LB_1,R_val=.5)

#Lasso with MA only

I_df_MA=MA(I_df_transform)
I_df_MA_post_2003=I_df_MA[~(I_df_MA.index<'2003-01-01')]


lasso_coef_MA={}
lasso_alpha_MA={}
lasso_model_MA={}
lasso_R2_MA={}
lasso_predict_MA={}
lasso_y_actual_MA={}
for y in D_df_post_2003.columns:
    print("Working on model for " + str(y))
    X=scaler.fit_transform(I_df_MA_post_2003)
    Y=D_df_post_2003[y]
    n=y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y = LassoCV(cv=5, random_state=42,alphas=[.025,.05,.075,.1,.2,.3,.4,.5,.7],max_iter=5000).fit(X_train, y_train)
    lasso_coef_MA[n]=y.coef_
    lasso_alpha_MA[n]=y.alpha_   
    lasso_model_MA[n]=y
    lasso_R2_MA[n]=y.score(X_test,y_test)
    lasso_predict_MA[n]=y.predict(X_test)
    lasso_y_actual_MA[n]=y_test

coef_df_MA=coef_DF(lasso_coef_MA,I_df_MA_post_2003,lasso_R2_MA,R_val=.5)


#plot predicted vs actual combo vars
for key in lasso_y_actual:
    j=key
    key=acc_plot(lasso_y_actual[key],lasso_predict[key],key,round(lasso_R2[key],3))
    key.savefig("plots/Predict_Actual/Combo/"+ str(j) + ".png")

#plot predicted vs actual no combo vars
for key in lasso_y_actual_TS:
    j=key
    key=acc_plot(lasso_y_actual_TS[key],lasso_predict_TS[key],key,round(lasso_R2_TS[key],3))
    key.savefig("plots/Predict_Actual/No_combo/"+ str(j) + ".png")
    
#plot predicted vs actual lag buckets
for key in lasso_y_actual_LB:
    j=key
    key=acc_plot(lasso_y_actual_LB[key],lasso_predict_LB[key],key,round(lasso_R2_LB[key],3))
    key.savefig("plots/Predict_Actual/Lag_buckets/"+ str(j) + ".png")
 
for key in lasso_y_actual_LB_1:
    j=key
    key=acc_plot(lasso_y_actual_LB_1[key],lasso_predict_LB_1[key],key,round(lasso_R2_LB_1[key],3))
    key.savefig("plots/Predict_Actual/lag_buckets_no_combo/"+ str(j) + ".png")
    
for key in lasso_y_actual_BL:
    j=key
    key=acc_plot(lasso_y_actual_BL[key],lasso_predict_BL[key],key,round(lasso_R2_BL[key],3))
    key.savefig("plots/Predict_Actual/Baseline/"+ str(j) + ".png")

#save csvs of coefs,r^2,alpha, for all models.
lasso_R2_df=pd.DataFrame([lasso_R2_BL,lasso_R2_MA,lasso_R2_LB_1,lasso_R2_TS,lasso_R2_LB,lasso_R2]).T
lasso_R2_df.columns=['Baseline','MA','MA_Lag_Buckets','MA_Lag_buckets_Combos','MA_Mthly_Lags','MA_Mthly_lags_Combos']

coef_df_MA.to_csv("lasso_ma_coefs.csv", index = True)
