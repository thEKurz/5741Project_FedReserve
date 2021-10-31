# -*- coding: utf-8 -*-
"""
MIDTERM ANALYSIS

Created on Fri Oct 29 14:40:49 2021

@author: evank
"""

import numpy as np
import pandas as pd
from IPython.display import Image
import matplotlib.pyplot as plt
#from folktables import ACSDataSource, ACSEmployment
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef
from statsmodels.graphics.tsaplots import plot_acf

Text_DF=pd.read_csv('text_sent_1.csv',index_col='Date',parse_dates=['Date'])
I_df=pd.read_csv('Independant_Numeric_Variables.csv',index_col='DATE',parse_dates=['DATE'])
D_df=pd.read_csv('Dependant_Numeric_Variables.csv',index_col='DATE',parse_dates=['DATE'])

#Text_DF.set_index(pd.to_datetime(Text_DF.index), inplace=True)

I_df[Text_DF.columns]=Text_DF[Text_DF.columns]
I_df.fillna(method='ffill',inplace=True)
I_df.fillna(0,inplace=True)

I_df_post_2003=I_df[~(I_df.index<'2003-01-01')]
D_df_post_2003=D_df[~(D_df.index<'2003-01-01')]


I_Corr_matrix=I_df_post_2003.corr(method='pearson')
D_Corr_matrix=D_df_post_2003.corr(method='pearson')

