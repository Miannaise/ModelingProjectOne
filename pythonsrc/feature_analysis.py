import seaborn as sns
import numpy as np
from train import X_train, Y_non_violent, Y_violent
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
    f.read()

#Avoid h(x)=1, start from real features
cov_arr=[]
for col in range(1,78):
    cov_arr.append(covariance(X_train[:,col],Y_violent))

top_five=cov_arr[-5:]
for i in top_five:
    print(i)