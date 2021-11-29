import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt


def get_coefs(coef_values,DF):
    coef_list=list(DF.columns)
    coef_df=pd.DataFrame(coef_values,index=coef_list)
    coef_df=coef_df.loc[~(coef_df==0).all(axis=1)]
    return coef_df

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

def TS_pipe(X,window_size=24):
    df=X
    df_1=df
    for window in range(1, window_size + 1):
        shifted = df_1.shift(window)
        df=df.join(shifted.rename(columns=lambda x: x+ "_" + str(window) + "_lag"))
    for x in df_1.columns:
        df[str(x) + '_MA_' + str(window_size)] = df[x].rolling(window=window_size).mean()
    return df

I_df_post_2003=I_df_transform[~(I_df_transform.index<'2003-01-01')]
D_df_post_2003=D_df[~(D_df.index<'2003-01-01')]

I_df_post_2003.to_csv("Independent_post2003.csv", index = True)
D_df_post_2003.to_csv("Dependent_post2003.csv", index = True)

I_df_pipe = TS_pipe(I_df_transform, 24)
I_df_pipe =I_df_pipe[~(I_df_pipe.index<'2003-01-01')]
I_df_pipe.to_csv("Independent_post2003_withlag.csv", index = True) 
