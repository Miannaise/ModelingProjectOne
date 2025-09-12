import seaborn as sns
import numpy as np
from train import X_train, Y_non_violent, Y_violent
import matplotlib.pyplot as plt
import typing

#We start with covariance since it is trivial to analyze
def covariance(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"Shapes do not match x.shape:{x.shape}, y.shape:{y.shape}")
    return np.mean((x - x.mean()) * (y - y.mean()))

#Read columns from columns.txt for easy callback.
with open("./proj1files/columns.txt",'r') as f:
    col_names=f.read().splitlines()

#Avoid h(x)=1, start from real features
cov_arr=[]
for col in range(1,78):
    cov_arr.append([covariance(X_train[:,col],Y_violent),col])

cov_arr.sort(key=lambda x: abs(x[0]),reverse=1)
top_five=cov_arr[:5]

#Top five features.
for i in range(0,5):  
    print(f"feature:{col_names[cov_arr[i][1]]} | cov: {cov_arr[i][0]}")

for cov, idx in top_five:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=X_train[:, idx+1], y=Y_violent)
    plt.xlabel(f"Feature {col_names[idx]}")
    plt.ylabel("Y_violent")
    plt.title(f"Scatter Plot: Feature vs Y_violent\nCovariance: {cov:.2f}")
    plt.tight_layout()
    plt.savefig(f"./media/plot{idx}.png")