import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

I_df = pd.read_csv("Independent_post2003.csv", index_col = "DATE", parse_dates = ["DATE"])
D_df = pd.read_csv("Dependent_post2003.csv", index_col = "DATE", parse_dates = ["DATE"])

# Get k-fold cross validation spilt
# k = 5
kf = sklearn.model_selection.KFold(n_splits=5)
kf.get_n_splits(I_df)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
i = 5
j = 0.1
crit = ['squared_error', 'absolute_error', 'poisson']
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


score_table = np.empty()
for d in range(D_df.shape[1]):
    y = D_df.iloc[:,[d,]]
    
    for train_index, test_index in kf.split(I_df):
        X_train = np.asarray(I_df.iloc[train_index,:])
        X_test = np.asarray(I_df.iloc[test_index,:])
        y_train = np.asarray(y.iloc[train_index,:])
        y_test = np.asarray(y.iloc[test_index,:])
        
        m1 = clf.fit(X_train, y_train)
