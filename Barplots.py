# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:34:13 2021

@author: evank

"""

figure, axis = plt.subplots(2, 2)
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
    
plt.figure()
plt.scatter(lasso_alpha_MA.keys(),lasso_alpha_MA.values())
plt.title('Alpha Parameter For each Lasso Model')
plt.xlabel('Dependent Variable')
plt.xticks(rotation=90)
plt.ylabel('Value')
plt.savefig("plots/Alpha_plot.png")