# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
import seaborn as sns
import statsmodels as sm

import scipy


from tsfresh import feature_selection, extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
from tsfresh.feature_extraction import ComprehensiveFCParameters, settings
from tsfresh.feature_selection.relevance import calculate_relevance_table

I_df=pd.read_csv('Independant_Numeric_Variables.csv',index_col='DATE',parse_dates=['DATE'])
D_df=pd.read_csv('Dependant_Numeric_Variables.csv',index_col='DATE',parse_dates=['DATE'])
I_df.fillna(0,inplace=True)
I_df.head()

I_df_post_2003=I_df[~(I_df.index<'2003-01-01')]
D_df_post_2003=D_df[~(D_df.index<'2003-01-01')]


# X_extracted = extract_features(I_df, column_id = )
X_selected = feature_selection.selection.select_features(I_df_post_2003, D_df_post_2003['GDP'])

print(X_selected)

I_df_post_2003.reset_index(inplace=True)
D_df_post_2003.reset_index(inplace=True)
I_df_post_2003.head()

X_extracted = extract_features(I_df_post_2003, column_id='DATE')
X_selected = select_features(X_extracted, D_df_post_2003['GDP'])
print(X_selected)


extracted_features = extract_features(
    I_df_post_2003,
    column_id="DATE",
)
extracted_features.fillna(0, inplace=True)
relevance_table = calculate_relevance_table(extracted_features, D_df_post_2003['GDP'])
relevance_table = relevance_table[relevance_table.relevant]
relevance_table.sort_values("p_value", inplace=True)
print(relevance_table)


fedgdppval = feature_selection.significance_tests.target_real_feature_real_test(I_df_post_2003['FEDFUNDS'], D_df_post_2003['GDP'])
print(fedgdppval)



