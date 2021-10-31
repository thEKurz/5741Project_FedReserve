# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import glob
files = glob.glob("CSV_Data/*.csv")

dflist=[]

#make list of dataframes
for f in files:
    csv = pd.read_csv(f)
    dflist.append(csv)
    
#make our dataframe 
Numerical_data_df= pd.DataFrame()
Arith=['FEDDT','FEDD10Y','RESPPNTNWW', 'M1SL','TREAST','TREAS10Y','TREAS1T5', 'TREAS5T10', 'TREAS911Y', 'MBS10Y','WFRBST01134']
Perc=['SPY','GDP','WILL5000INDFC', 'WILLSMLCAP','CSUSHPINSA']

dflist_aft = dflist
for n in range(0,len(dflist)):
    #standardize dates and make datetime index
    dflist[n].iloc[:,0] = dflist[n].iloc[:,0].str.replace('/','-')
    dflist[n].rename(columns={ dflist[n].columns[0]: "DATE" }, inplace = True)
    dflist[n]['DATE']=pd.to_datetime(dflist[n]['DATE'])
    dflist[n].set_index(dflist[n].iloc[:,0], inplace=True)
    dflist[n]=dflist[n].drop(dflist[n].columns[[0]], axis=1)
    
    #remove rows with non-numeric data
    dflist[n]=dflist[n][pd.to_numeric(dflist[n].iloc[:,0], errors='coerce').notnull()]
    dflist[n].iloc[:,0]=pd.to_numeric(dflist[n].iloc[:,0])
    if dflist[n].columns[0] in Perc:
        dflist[n]= (dflist[n].shift(-1)/dflist[n])-1
    
    #merge frame without losing any data
    Numerical_data_df= Numerical_data_df.merge(dflist[n], how='outer', left_index=True, right_index=True)

Numerical_data_df.fillna(method='ffill',inplace=True)
Numerical_data_df.fillna(0,inplace=True)
Freq_data_df=Numerical_data_df.groupby(Numerical_data_df.index.to_period('M')).nth(0)

#columns that need to be differenced for returns

Freq_data_df[Arith]=Freq_data_df[Arith].shift(-1) - Freq_data_df[Arith]

I_var=['FEDFUNDS','FEDDT','FEDD10Y','RESPPNTNWW', 'M1SL','TREAST','TREAS10Y','TREAS1T5', 'TREAS5T10', 'TREAS911Y', 'MBS10Y']
I_df=Freq_data_df[I_var]
D_var=list(Freq_data_df.columns.difference(I_var).astype(str))
D_df = Freq_data_df[Freq_data_df.columns.difference(I_var)]
D_df.to_csv('Dependant_Numeric_Variables.csv',index=True)
I_df.to_csv('Independant_Numeric_Variables.csv',index=True)