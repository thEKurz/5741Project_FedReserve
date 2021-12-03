import pandas as pd
import numpy as np
import math

# Trim number of dependent variables and map 1 to non-zero coefficients
coef_table = pd.read_csv("lasso_ma_coefs.csv", index_col = 0)
cl = coef_table.columns
print(cl)

cl = cl[[0,2,3,4,7,9,10]]
coef_table = coef_table[cl]

def replace(x):
    if math.isnan(x):
        return 0
    else:
        return 1

coef_ind = coef_table.applymap(lambda x: replace(x))

# Plot grouped barplot
import matplotlib.pyplot as plt

label_norm = list()
label_ma = list()

for label in coef_ind.index:
    if "_MA_24" in label:
        label_ma.append(label)
    else:
        label_norm.append(label)

label_norm.sort()
label_ma.sort()

label_norm.insert(0, "FEDD10Y")
label_norm.insert(3, "M1SL")

bars_norm = list()
bars_ma = list()

for label in label_norm:
    if label in coef_ind.index:
        bars_norm.append(np.sum(coef_ind.loc[label,:]))
    else:
        bars_norm.append(0)

for label in label_ma:
    if label in coef_ind.index:
        bars_ma.append(np.sum(coef_ind.loc[label,:]))
    else:
        bars_ma.append(0)

barWidth = 0.25
r1 = np.arange(len(bars_norm))
r2 = [x + barWidth for x in r1]

plt.bar(r1, bars_norm, color='b', width=barWidth, edgecolor='white', label='Original')
plt.bar(r2, bars_ma, color='r', width=barWidth, edgecolor='white', label='Moving Average')
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars_norm))], label_norm)
plt.xticks(rotation = 90)
plt.title("Number of dependent varibles affected per independent variables")
plt.legend()
plt.show()