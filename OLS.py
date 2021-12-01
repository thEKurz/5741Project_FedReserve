import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
import matplotlib.pyplot as plt

I_df = pd.read_csv("Independent_post2003.csv", index_col = "DATE", parse_dates = ["DATE"])
D_df = pd.read_csv("Dependent_post2003.csv", index_col = "DATE", parse_dates = ["DATE"])
   
# Get k-fold cross validation spilt
# k = 5
kf = sklearn.model_selection.KFold(n_splits=5)
kf.get_n_splits(I_df)
    
# Linear regression OLS
from sklearn.linear_model import LinearRegression

def linearOLS(I_df, D_df):
    avg_score_table = pd.DataFrame(index = D_df.columns, columns = ["Score"])
        
    for d in range(D_df.shape[1]):
        y = D_df.iloc[:,d]
        model = LinearRegression(fit_intercept=True)
        n = 0
        score_table = np.empty(5)
        for train_index, test_index in kf.split(I_df):
            X_train = np.asarray(I_df.iloc[train_index,:])
            X_test = np.asarray(I_df.iloc[test_index,:])
            y_train = np.asarray(y.iloc[train_index])
            y_test = np.asarray(y.iloc[test_index])
                            
            m1 = model.fit(X_train, y_train)
            score_table[n] = m1.score(X_test, y_test)
            n += 1
                
        avg_score_table.loc[D_df.columns[d], 'Score'] = np.mean(score_table)
    
    return avg_score_table


avg_normal = linearOLS(I_df, D_df)

I_df = pd.read_csv("Independent_post2003_withlag.csv", index_col = "DATE", parse_dates = ["DATE"])
avg_lags = linearOLS(I_df, D_df)

I_df = pd.read_csv("Independent_post2003_buckets.csv", index_col = "DATE", parse_dates = ["DATE"])
avg_buckets = linearOLS(I_df, D_df)

plt.scatter(avg_normal.index, avg_normal, label = "Original data")
plt.scatter(avg_normal.index, avg_lags, label = "With lags")
plt.scatter(avg_normal.index, avg_buckets, label = "With moving average buckets")
plt.legend()
plt.ylim([-1000,0])
plt.xticks(rotation = 90)
plt.title("Linear OLS with various manipulation of data")
plt.show()

