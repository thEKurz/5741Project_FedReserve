import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_text = pd.read_csv("text_sent_1.csv", index_col = "Date")

df_text_stats = pd.DataFrame(columns = df_text.columns, index = ['mean', 'std'])
for i in df_text.columns:
    df_text_stats.loc['mean',i] = round(np.mean(df_text[i]),4)
    df_text_stats.loc['std',i] = round(np.std(df_text[i]),4)
print(df_text_stats)   
    
for i in df_text.columns:
    plt.plot(df_text.index, df_text[i], label = i)
plt.legend()
plt.title("Plot of sentiment change over time")
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.show()




from matplotlib.gridspec import GridSpec

df_ind = pd.read_csv("Independant_Numeric_Variables.csv", index_col = "DATE")
df_ind = df_ind[~(df_ind.index <= '2003-01-01')]

df_ind_stats = pd.DataFrame(columns = df_ind.columns, index = ['mean', 'std'])

for i in df_ind.columns:
    df_ind_stats.loc['mean',i] = round(np.mean(df_ind[i]),4)
    df_ind_stats.loc['std',i] = round(np.std(df_ind[i]),4)
    plt.hist(df_ind[i], bins = 100)
    plt.title("Histogram of "+ i)
    plt.show()

print(df_ind_stats)
    