import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
import matplotlib.pyplot as plt

I_df = pd.read_csv("Independent_post2003_withlag.csv", index_col = "DATE", parse_dates = ["DATE"])
D_df = pd.read_csv("Dependent_post2003.csv", index_col = "DATE", parse_dates = ["DATE"])

# Get k-fold cross validation spilt
# k = 5
kf = sklearn.model_selection.KFold(n_splits=5)
kf.get_n_splits(I_df)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# For one dependent variable
results = list()

for d in range(D_df.shape[1]):
    y = D_df.iloc[:,d]
    
    max_depth = np.arange(5,18,1)
    min_samples_split = np.arange(0.01,0.1,0.01)
    crit = ['squared_error', 'absolute_error', 'poisson']
    sum_score_table = pd.DataFrame(index = max_depth, columns = min_samples_split)
    for i in max_depth:
        for j in min_samples_split:
            
            clf = RandomForestRegressor(n_estimators=100, 
                                         criterion=crit[0], 
                                         max_depth=i, 
                                         min_samples_split=j, 
                                         max_features='auto', 
                                         max_leaf_nodes=None, 
                                         bootstrap=True, 
                                         n_jobs=4, 
                                         random_state=2, 
                                         max_samples=None)
            
            n = 0
            score_table = np.empty(5)
            for train_index, test_index in kf.split(I_df):
                X_train = np.asarray(I_df.iloc[train_index,:])
                X_test = np.asarray(I_df.iloc[test_index,:])
                y_train = np.asarray(y.iloc[train_index])
                y_test = np.asarray(y.iloc[test_index])
                    
                m1 = clf.fit(X_train, y_train)
                score_table[n] = m1.score(X_test, y_test)
                n += 1
   
            sum_score_table.loc[i,j] = np.sum(score_table)
    results.append(sum_score_table)
    plt.scatter(y_test, m1.predict(X_test))
    plt.show() 


path = r"C:\Users\Kelvin\Desktop\ORIE5741_learning_with_big_messy_data\big_messy_project\5741Project_FedReserve\plots\random_forest"
for d in range(D_df.shape[1]):
    results[0].to_csv(D_df.columns[d]+"_model_selection_score.csv", index = True)