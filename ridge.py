import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
import matplotlib.pyplot as plt

I_df = pd.read_csv("Independent_post2003.csv", index_col = "DATE", parse_dates = ["DATE"])
D_df = pd.read_csv("Dependent_post2003.csv", index_col = "DATE", parse_dates = ["DATE"])

def perform_ridge(I_df, D_df):
    # Get k-fold cross validation spilt
    # k = 5
    kf = sklearn.model_selection.KFold(n_splits=5)
    kf.get_n_splits(I_df)
    
    # Prepare for ridge regression
    from sklearn.linear_model import Ridge
    
    
    regularizer_coef = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
    avg_score_table = pd.DataFrame(index = D_df.columns, columns = regularizer_coef)
    
    for d in range(D_df.shape[1]):
        y = D_df.iloc[:,d]
        
        for i in regularizer_coef:
            rid = Ridge(alpha= i, 
                fit_intercept = True, 
                normalize = True, 
                tol=0.001)
            
            n = 0
            score_table = np.empty(5)
            for train_index, test_index in kf.split(I_df):
                X_train = np.asarray(I_df.iloc[train_index,:])
                X_test = np.asarray(I_df.iloc[test_index,:])
                y_train = np.asarray(y.iloc[train_index])
                y_test = np.asarray(y.iloc[test_index])
                        
                m1 = rid.fit(X_train, y_train)
                score_table[n] = m1.score(X_test, y_test)
                n += 1
            
            avg_score_table.loc[D_df.columns[d],i] = np.mean(score_table)
    return avg_score_table

avg_normal = perform_ridge(I_df, D_df)

I_df = pd.read_csv("Independent_post2003_withlag.csv", index_col = "DATE", parse_dates = ["DATE"])
avg_lags = perform_ridge(I_df, D_df)

I_df = pd.read_csv("Independent_post2003_buckets.csv", index_col = "DATE", parse_dates = ["DATE"])
avg_buckets = perform_ridge(I_df, D_df)

for d in range(D_df.shape[1]):
    plt.plot(avg_normal.columns, avg_normal.iloc[d,:], label = avg_normal.index[d])
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Out of sample prediction score for ridge regression")
plt.show()

for d in range(D_df.shape[1]):
    plt.plot(avg_lags.columns, avg_lags.iloc[d,:], label = avg_lags.index[d])
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Out of sample prediction score for ridge regression with lags")
plt.show()

for d in range(D_df.shape[1]):
    plt.plot(avg_buckets.columns, avg_buckets.iloc[d,:], label = avg_buckets.index[d])
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Out of sample prediction score for ridge regression with moving average buckets")
plt.show()