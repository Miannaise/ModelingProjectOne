import scipy.io as scio
import numpy as np
#Load the matlab file - Mia
training_data=scio.loadmat("./proj1files/trainingdata.mat")
#Weird Dict, one key being "trainingdata" and the value is an np.ndarray
#Extract that out of that
training_data=training_data["trainingdata"]
#print(type(training_data))
#X data
X_train=training_data[:,0:78]
print(np.shape(X_train)) 
#violent crimes.
Y_violent=training_data[:,79]
print(np.shape(Y_violent))
#non violent crime
b_iso=[0]*78
Y_non_violent=training_data[:,80]
j=0
# for i in X_train.T:
#     #print(i.shape)
#     X = np.append(np.ones(i.shape)[:,np.newaxis], i[:,np.newaxis], axis=1)
#     #print(X.shape)
#     # X = np.hstack((np.ones((i.shape[0], 1)), i))
#     b_iso[j]=np.linalg.lstsq(X,Y_non_violent)
#     #print(b_iso[j])
#     j+=1
import numpy as np

def fit_a_over_x_plus_b(x, y):
    # 1) Build regressor z = 1/x and mask invalid values
    z = 1.0 / x

    # 2) Build response matrix [z, 1]
    A = np.column_stack([z, np.ones_like(z)])

    # 3) Least squares solution
    # Solve A @ [a, b] ≈ y
    params, residuals, rank, svals = np.linalg.lstsq(A, y, rcond=None)
    a, b = params

    # 4) Diagnostics
    y_hat = A @ params
    rss = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - rss / tss if tss > 0 else np.nan

    return a, b, {"rss": rss, "r2": r2, "residuals": residuals, "rank": rank, "svals": svals}

# Example


a, b, info = fit_a_over_x_plus_b((X_train.T)[13],Y_violent)
print(f"a = {a:.6f}, b = {b:.6f}, R^2 = {info['r2']:.4f}")
