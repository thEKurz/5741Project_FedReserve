# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 22:48:28 2021

@author: evank
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 12:18:24 2021

@author: evank
"""


I_df_drop_totals=I_df_transform.drop(['FEDD10Y','RESPPNTNWW','TREAST','Neutral'],axis=1)


I_df_drop_post_2003=I_df_drop_totals[~(I_df_drop_totals.index<'2003-01-01')]
D_df_post_2003=D_df[~(D_df.index<'2003-01-01')]




from sklearn.model_selection import train_test_split


#linear regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

#lasso baseline
lasso1_coef_BL={}
lasso1_alpha_BL={}
lasso1_model_BL={}
lasso1_R2_BL={}
lasso1_predict_BL={}
lasso1_y_actual_BL={}
for y in D_df_post_2003.columns:
    print("Working on model for " + str(y))
    X=scaler.fit_transform(I_df_drop_post_2003)
    Y=D_df_post_2003[y]
    n=y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y = LassoCV(cv=5, random_state=42,alphas=[.025,.05,.075,.1,.2,.3,.4,.5,.7],max_iter=5000).fit(X_train, y_train)
    lasso1_coef_BL[n]=y.coef_
    lasso1_alpha_BL[n]=y.alpha_   
    lasso1_model_BL[n]=y
    lasso1_R2_BL[n]=y.score(X_test,y_test)
    lasso1_predict_BL[n]=y.predict(X_test)
    lasso1_y_actual_BL[n]=y_test

coef_df1_BL=coef_DF(lasso1_coef_BL,I_df_drop_post_2003,lasso1_R2_BL,R_val=.5)


I_df_linear_reg=TS_pipe(variable_combos(I_df_drop_totals))
I_df_LR_post_2003=I_df_linear_reg[~(I_df_linear_reg.index<'2003-01-01')]

lasso1_coef={}
lasso1_alpha={}
lasso1_model={}
lasso1_R2={}
lasso1_predict={}
lasso1_y_actual={}
for y in D_df_post_2003.columns:
    print("Working on model for " + str(y))
    X=scaler.fit_transform(I_df_LR_post_2003)
    Y=D_df_post_2003[y]
    n=y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y = LassoCV(cv=5, random_state=42,alphas=[.025,.05,.075,.1,.2,.3,.4,.5,.7],max_iter=5000).fit(X_train, y_train)
    lasso1_coef[n]=y.coef_
    lasso1_alpha[n]=y.alpha_   
    lasso1_model[n]=y
    lasso1_R2[n]=y.score(X_test,y_test)
    lasso1_predict[n]=y.predict(X_test)
    lasso1_y_actual[n]=y_test
    

#get coefficients from models with R^2 >.5
coef_df1=coef_DF(lasso1_coef,I_df_LR_post_2003,lasso1_R2,R_val=.5)


    
#Lasso without combined features
I_df_TS=TS_pipe(I_df_drop_totals)
I_df_TS_post_2003=I_df_TS[~(I_df_TS.index<'2003-01-01')]

lasso1_coef_TS={}
lasso1_alpha_TS={}
lasso1_model_TS={}
lasso1_R2_TS={}
lasso1_predict_TS={}
lasso1_y_actual_TS={}
for y in D_df_post_2003.columns:
    print("Working on model for " + str(y))
    X=scaler.fit_transform(I_df_TS_post_2003)
    Y=D_df_post_2003[y]
    n=y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y = LassoCV(cv=5, random_state=42,alphas=[.025,.05,.075,.1,.2,.3,.4,.5,.7],max_iter=5000).fit(X_train, y_train)
    lasso1_coef_TS[n]=y.coef_
    lasso1_alpha_TS[n]=y.alpha_   
    lasso1_model_TS[n]=y
    lasso1_R2_TS[n]=y.score(X_test,y_test)
    lasso1_predict_TS[n]=y.predict(X_test)
    lasso1_y_actual_TS[n]=y_test

#get coefficients from models with R^2 >.5
coef_df1_TS=coef_DF(lasso1_coef_TS,I_df_TS_post_2003,lasso1_R2_TS,R_val=.5)


#Lasso with lag buckets and cobmined featured

I_df_lag_buckets=TS_lag_buckets(variable_combos(I_df_drop_totals))
I_df_LB_post_2003=I_df_lag_buckets[~(I_df_lag_buckets.index<'2003-01-01')]

lasso1_coef_LB={}
lasso1_alpha_LB={}
lasso1_model_LB={}
lasso1_R2_LB={}
lasso1_predict_LB={}
lasso1_y_actual_LB={}
for y in D_df_post_2003.columns:
    print("Working on model for " + str(y))
    X=scaler.fit_transform(I_df_LB_post_2003)
    Y=D_df_post_2003[y]
    n=y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y = LassoCV(cv=5, random_state=42,alphas=[.025,.05,.075,.1,.2,.3,.4,.5,.7],max_iter=5000).fit(X_train, y_train)
    lasso1_coef_LB[n]=y.coef_
    lasso1_alpha_LB[n]=y.alpha_   
    lasso1_model_LB[n]=y
    lasso1_R2_LB[n]=y.score(X_test,y_test)
    lasso1_predict_LB[n]=y.predict(X_test)
    lasso1_y_actual_LB[n]=y_test

coef_df1_LB=coef_DF(lasso1_coef_LB,I_df_LB_post_2003,lasso1_R2_LB,R_val=.5)


#Lasso with lag buckets only

I_df_LB=TS_lag_buckets(I_df_drop_totals)
I_df_LB_post_2003_1=I_df_LB[~(I_df_LB.index<'2003-01-01')]

lasso1_coef_LB_1={}
lasso1_alpha_LB_1={}
lasso1_model_LB_1={}
lasso1_R2_LB_1={}
lasso1_predict_LB_1={}
lasso1_y_actual_LB_1={}
for y in D_df_post_2003.columns:
    print("Working on model for " + str(y))
    X=scaler.fit_transform(I_df_LB_post_2003_1)
    Y=D_df_post_2003[y]
    n=y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y = LassoCV(cv=5, random_state=42,alphas=[.025,.05,.075,.1,.2,.3,.4,.5,.7],max_iter=5000).fit(X_train, y_train)
    lasso1_coef_LB_1[n]=y.coef_
    lasso1_alpha_LB_1[n]=y.alpha_   
    lasso1_model_LB_1[n]=y
    lasso1_R2_LB_1[n]=y.score(X_test,y_test)
    lasso1_predict_LB_1[n]=y.predict(X_test)
    lasso1_y_actual_LB_1[n]=y_test

coef_df1_LB_1=coef_DF(lasso1_coef_LB_1,I_df_LB_post_2003_1,lasso1_R2_LB_1,R_val=.5)

#Lasso with MA only

I_df_MA=MA(I_df_drop_totals)
I_df_MA_post_2003=I_df_MA[~(I_df_LB.index<'2003-01-01')]

lasso1_coef_MA={}
lasso1_alpha_MA={}
lasso1_model_MA={}
lasso1_R2_MA={}
lasso1_predict_MA={}
lasso1_y_actual_MA={}
for y in D_df_post_2003.columns:
    print("Working on model for " + str(y))
    X=scaler.fit_transform(I_df_MA_post_2003)
    Y=D_df_post_2003[y]
    n=y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    y = LassoCV(cv=5, random_state=42,alphas=[.025,.05,.075,.1,.2,.3,.4,.5,.7],max_iter=5000).fit(X_train, y_train)
    lasso1_coef_MA[n]=y.coef_
    lasso1_alpha_MA[n]=y.alpha_   
    lasso1_model_MA[n]=y
    lasso1_R2_MA[n]=y.score(X_test,y_test)
    lasso1_predict_MA[n]=y.predict(X_test)
    lasso1_y_actual_MA[n]=y_test

coef_df1_MA=coef_DF(lasso1_coef_MA,I_df_MA_post_2003,lasso1_R2_MA,R_val=.5)


#save csvs of coefs,r^2,alpha, for all models.
lasso1_R2_df=pd.DataFrame([lasso1_R2_BL,lasso1_R2_LB_1,lasso1_R2_TS,lasso1_R2_LB,lasso1_R2]).T
lasso1_R2_df.columns=['Baseline','MA_Lag_Buckets','MA_Lag_buckets_Combos','MA_Mthly_Lags','MA_Mthly_lags_Combos']

