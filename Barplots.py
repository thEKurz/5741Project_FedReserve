# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:34:13 2021

@author: evank
"""
for p in coef_df_MA.columns:
    y= coef_df_MA.loc[:,p].dropna()
    x=y.index.tolist()
    plt.figure()
    plt.bar(x,y,color='g')
    plt.title('Top Model Coefficients For ' + str(p))
    plt.xlabel('Coefficient')
    plt.xticks(rotation=90)
    plt.ylabel('Value')
    plt.savefig("plots/Coeficient_MA_plots/coef_"+ str(p) + ".png")