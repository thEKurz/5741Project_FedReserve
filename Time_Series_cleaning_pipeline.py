# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import glob
files = glob.glob("C:/Users/evank/OneDrive/Documents/Academics/Cornell/Fall 2021/5741 Learning With Big Messy Data/Project/5741Project_FedReserve/CSV_Data/*.csv")

dflist=[]

#make list of dataframes
for f in files:
    csv = pd.read_csv(f)
    dflist.append(csv)
    
#make our dataframe 
Numerical_data_df= pd.DataFrame()

for n in range(0,len(dflist)):
    #standardize dates and make datetime index
    dflist[n].iloc[:,0] = dflist[n].iloc[:,0].str.replace('/','-')
    dflist[n].rename(columns={ dflist[n].columns[0]: "DATE" }, inplace = True)
    dflist[n]['DATE']=pd.to_datetime(dflist[n]['DATE'])
    dflist[n].set_index(dflist[n].iloc[:,0], inplace=True)
    dflist[n]=dflist[n].drop(dflist[n].columns[[0]], axis=1)
    
    #remove rows with non-numeric data
    dflist[n]=dflist[n][pd.to_numeric(dflist[n].iloc[:,0], errors='coerce').notnull()]
    
    #merge frame without losing any data
    Numerical_data_df= Numerical_data_df.merge(dflist[n], how='outer', left_index=True, right_index=True)

Numerical_data_df.fillna(method='ffill',inplace=True)
Numerical_data_df.fillna(0,inplace=True)
Freq_data_df=Numerical_data_df.groupby(Numerical_data_df.index.to_period('M')).nth(0)
I_var=['FEDFUNDS','FEDDT','FEDD10Y','RESPPNTNWW', 'M1SL','TREAST','TREAS10Y','TREAS1T5', 'TREAS5T10', 'TREAS911Y', 'MBS10Y']
I_df=Freq_data_df[I_var]
D_var=list(Freq_data_df.columns.difference(I_var).astype(str))
D_df = Freq_data_df[Freq_data_df.columns.difference(I_var)]
D_df.to_csv('Dependant_Numeric_Variables',index=True)
I_df.to_csv('Independant_Numeric_Variables',index=True)