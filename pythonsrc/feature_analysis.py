import seaborn as sns
import numpy as np
from train import X_train, Y_non_violent, Y_violent
import typing

for col in X_train:
    print(np.cov(X_train[col],Y_violent))