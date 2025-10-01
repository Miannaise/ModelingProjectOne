import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import makedirs, path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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
Y_non_violent=training_data[:,80]

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def normalize_feature(x, y):
    mean = np.mean(y,axis = 0,keepdims=True)
    std = np.std(y,axis = 0,keepdims=True)
    return (x - mean) / (std + 1e-8)

normalize_features = normalize_feature(X_train,X_train)
# print(np.shape(normalize_features))
# print(normalize_features.mean(axis = 0))
# print(normalize_features.var(axis = 0))
covar= np.cov(normalize_features,rowvar=False)

def covar_heatmap(X):
    plt.figure(figsize=(10, 8))
    cov_matrix = np.cov(X, rowvar=False)
    plt.imshow(cov_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Covariance')
    plt.title('Covariance Heatmap')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.tight_layout()
    plt.show()

#covar_heatmap(covar)
def correlation_heatmap(X):
    plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(X, rowvar=False)
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Correlation Heatmap')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.tight_layout()
    plt.show()
#correlation_heatmap(np.corrcoef(normalize_features,rowvar=False))

#Implement Dropout for highly correlated features
def dropout_highly_correlated_features(X, threshold=0.9):
    corr_matrix = np.corrcoef(X, rowvar=False)
    to_drop = set()
    num_features = corr_matrix.shape[0]

    for i in range(num_features):
        for j in range(i + 1, num_features):
            if abs(corr_matrix[i, j]) > threshold:
                to_drop.add(j)

    keep_indices = [i for i in range(num_features) if i not in to_drop]
    return X[:, keep_indices], keep_indices
X_dropped_normalized, kept_indices = dropout_highly_correlated_features(normalize_features, threshold=0.9)
print(kept_indices)
print(f"Shape after dropping highly correlated features: {X_dropped_normalized.shape}")
#Scatter plot for X_dropped_normalized for violent and non-violent
def scatter_plots(X, y_violent, y_non_violent, kept_indices):
    output_dir = "./media/normalized"
    if not path.exists(output_dir):
        makedirs(output_dir)

    for idx, feature_idx in enumerate(kept_indices):
        plt.figure(figsize=(12, 5))

        # Plot for violent crimes
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=X[:, idx], y=y_violent, color='red', s=3)
        plt.title(f'Feature {feature_idx} vs Violent Crimes')
        plt.xlabel(f'Feature {feature_idx}')
        plt.ylabel('Y_violent')

        # Plot for non-violent crimes
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=X[:, idx], y=y_non_violent, color='blue', s=3)
        plt.title(f'Feature {feature_idx} vs Non-Violent Crimes')
        plt.xlabel(f'Feature {feature_idx}')
        plt.ylabel('Y_non_violent')

        plt.tight_layout()
        plt.savefig(path.join(output_dir, f'feature_{feature_idx}_scatter.png'))
        plt.close()
#scatter_plots(X_dropped_normalized, Y_violent, Y_non_violent, kept_indices)

#Compute Kurtosis
def compute_kurtosis(X):
    n = X.shape[0]
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    kurtosis = np.sum(((X - mean) / (std + 1e-8))**4, axis=0) / n -3
    return kurtosis
kurtosis_vector = compute_kurtosis(X_dropped_normalized)
#print("Kurtosis of each feature after dropping highly correlated ones:")
# for i in range(len(kurtosis_vector)):
#     print(f"Feature {kept_indices[i]}: Kurtosis = {kurtosis_vector[i]}")

#Revisit this.
#drop features with high kurtosis
def drop_high_kurtosis_features(X, kurtosis_vector, threshold=10):
    to_drop = [i for i, k in enumerate(kurtosis_vector) if abs(k) > threshold]
    keep_indices = [i for i in range(X.shape[1]) if i not in to_drop]
    return X[:, keep_indices], keep_indices
X_final, final_kept_indices = drop_high_kurtosis_features(X_dropped_normalized, kurtosis_vector, threshold=10)
print(f"Final shape after dropping high kurtosis features: {X_final.shape}")

#naiive regression on X_final
X_final = np.insert(X_final, 0, np.ones((X_final.shape[0],)), axis=1) #add intercept
#violent
beta_hat_v_naive,*_=np.linalg.lstsq(X_final, Y_violent, rcond=None)
Y_violent_pred_naive=  X_final @ beta_hat_v_naive
mape_v_naive = mape(Y_violent, Y_violent_pred_naive)
#log fit
Y_violent_log_naive = np.log(np.clip(Y_violent, 1e-5, None))
beta_hat_v_log_naive,*_=np.linalg.lstsq(X_final, Y_violent_log_naive, rcond=None)
Y_violent_exp_pred_naive=  np.exp(X_final @ beta_hat_v_log_naive)
mape_v_exp_naive = mape(Y_violent, Y_violent_exp_pred_naive)
print("\n\nNaiive Regression Results:")
print(f"Violent Crime MAPE (direct) naive: {mape_v_naive:.2f}")
print(f"Violent Crime MAPE (log fit on y) naive: {mape_v_exp_naive:.2f}")
#non-violent
beta_hat_nv_naive,*_=np.linalg.lstsq(X_final, Y_non_violent, rcond=None)
Y_non_violent_pred_naive=  X_final @ beta_hat_nv_naive
mape_nv_naive = mape(Y_non_violent, Y_non_violent_pred_naive)
#log fit
Y_non_violent_log_naive = np.log(np.clip(Y_non_violent, 1e-5, None))
beta_hat_nv_log_naive,*_=np.linalg.lstsq(X_final, Y_non_violent_log_naive, rcond=None)
Y_non_violent_exp_pred_naive=  np.exp(X_final @ beta_hat_nv_log_naive)
mape_nv_exp_naive = mape(Y_non_violent, Y_non_violent_exp_pred_naive)
print(f"Non Violent Crime MAPE (direct) naive: {mape_nv_naive:.2f}")
print(f"Non Violent Crime MAPE (log fit on y) naive: {mape_nv_exp_naive:.2f}")


#test data
#load test data
testing_data=scio.loadmat("./proj1files/testingdata.mat")
testing_data=testing_data["testingdata"]
X_test=testing_data[:,0:78]
Y_violent_test=testing_data[:,79]
Y_non_violent_test=testing_data[:,80]

#Normalize X_test, make sure it's the same normalization as training.
X_test = normalize_feature(X_test,X_train)

#drop same features as training
X_test_dropped = X_test[:, kept_indices]
X_test_final = X_test_dropped[:, final_kept_indices]
X_test_final = np.insert(X_test_final, 0, np.ones((X_test_final.shape[0],)), axis=1) #add intercept

#Violent
Y_violent_test_pred_naive=  X_test_final @ beta_hat_v_naive
mape_v_test_naive = mape(Y_violent_test, Y_violent_test_pred_naive)
Y_violent_test_exp_pred_naive=  np.exp(X_test_final @ beta_hat_v_log_naive)
mape_v_test_exp_naive = mape(Y_violent_test, Y_violent_test_exp_pred_naive)
print(f"Violent Crime Test MAPE (direct) naive: {mape_v_test_naive:.2f}")
print(f"Violent Crime Test MAPE (log fit on y) naive: {mape_v_test_exp_naive:.2f}")

#non-violent
Y_non_violent_test_pred_naive=  X_test_final @ beta_hat_nv_naive
mape_nv_test_naive = mape(Y_non_violent_test, Y_non_violent_test_pred_naive)
Y_non_violent_test_exp_pred_naive=  np.exp(X_test_final @ beta_hat_nv_log_naive)
mape_nv_test_exp_naive = mape(Y_non_violent_test, Y_non_violent_test_exp_pred_naive)
print(f"Non Violent Crime Test MAPE (direct) naive: {mape_nv_test_naive:.2f}")
print(f"Non Violent Crime Test MAPE (log fit on y) naive: {mape_nv_test_exp_naive:.2f}")



# L2 Regularization (Ridge Regression)
def ridge_regression(X, y, lmbda):
    n_features = X.shape[1]
    # Do not regularize the intercept (first column)
    I = np.eye(n_features)
    I[0, 0] = 0
    beta_ridge = np.linalg.inv(X.T @ X + lmbda * I) @ X.T @ y
    return beta_ridge

# Set regularization strength (lambda)
lmbda = 10  # You can tune this value

# Violent
beta_hat_v_ridge = ridge_regression(X_final, Y_violent, lmbda)
Y_violent_pred_ridge = X_final @ beta_hat_v_ridge
mape_v_ridge = mape(Y_violent, Y_violent_pred_ridge)

# Log fit
beta_hat_v_log_ridge = ridge_regression(X_final, Y_violent_log_naive, lmbda)
Y_violent_exp_pred_ridge = np.exp(X_final @ beta_hat_v_log_ridge)
mape_v_exp_ridge = mape(Y_violent, Y_violent_exp_pred_ridge)
print("\n\nRidge Regression Results:")
print(f"Violent Crime MAPE (direct) ridge: {mape_v_ridge:.2f}")
print(f"Violent Crime MAPE (log fit on y) ridge: {mape_v_exp_ridge:.2f}")

# Non-violent
beta_hat_nv_ridge = ridge_regression(X_final, Y_non_violent, lmbda)
Y_non_violent_pred_ridge = X_final @ beta_hat_nv_ridge
mape_nv_ridge = mape(Y_non_violent, Y_non_violent_pred_ridge)

# Log fit
beta_hat_nv_log_ridge = ridge_regression(X_final, Y_non_violent_log_naive, lmbda)
Y_non_violent_exp_pred_ridge = np.exp(X_final @ beta_hat_nv_log_ridge)
mape_nv_exp_ridge = mape(Y_non_violent, Y_non_violent_exp_pred_ridge)

print(f"Non Violent Crime MAPE (direct) ridge: {mape_nv_ridge:.2f}")
print(f"Non Violent Crime MAPE (log fit on y) ridge: {mape_nv_exp_ridge:.2f}")

# Test set predictions with ridge
Y_violent_test_pred_ridge = X_test_final @ beta_hat_v_ridge
mape_v_test_ridge = mape(Y_violent_test, Y_violent_test_pred_ridge)
Y_violent_test_exp_pred_ridge = np.exp(X_test_final @ beta_hat_v_log_ridge)
mape_v_test_exp_ridge = mape(Y_violent_test, Y_violent_test_exp_pred_ridge)
print(f"Violent Crime Test MAPE (direct) ridge: {mape_v_test_ridge:.2f}")
print(f"Violent Crime Test MAPE (log fit on y) ridge: {mape_v_test_exp_ridge:.2f}")

Y_non_violent_test_pred_ridge = X_test_final @ beta_hat_nv_ridge
mape_nv_test_ridge = mape(Y_non_violent_test, Y_non_violent_test_pred_ridge)
Y_non_violent_test_exp_pred_ridge = np.exp(X_test_final @ beta_hat_nv_log_ridge)
mape_nv_test_exp_ridge = mape(Y_non_violent_test, Y_non_violent_test_exp_pred_ridge)
print(f"Non Violent Crime Test MAPE (direct) ridge: {mape_nv_test_ridge:.2f}")
print(f"Non Violent Crime Test MAPE (log fit on y) ridge: {mape_nv_test_exp_ridge:.2f}")



# L1 Regularization (Lasso Regression) from scratch using coordinate descent
def soft_thresholding(rho, lmbda):
    if rho < -lmbda:
        return rho + lmbda
    elif rho > lmbda:
        return rho - lmbda
    else:
        return 0.0

def lasso_coordinate_descent(X, y, lmbda, num_iters=1000, tol=1e-4):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    for it in range(num_iters):
        w_old = w.copy()
        for j in range(n_features):
            X_j = X[:, j]
            y_pred = X @ w
            # Remove contribution of w[j]
            residual = y - y_pred + w[j] * X_j
            rho = X_j @ residual
            if j == 0:
                # Do not regularize intercept
                w[j] = rho / (X_j @ X_j)
            else:
                w[j] = soft_thresholding(rho, lmbda / 2) / (X_j @ X_j)
        if np.linalg.norm(w - w_old, ord=1) < tol:
            break
    return w

lmbda_lasso = 0.1  # You can tune this value

# Violent
beta_hat_v_lasso = lasso_coordinate_descent(X_final, Y_violent, lmbda_lasso)
Y_violent_pred_lasso = X_final @ beta_hat_v_lasso
mape_v_lasso = mape(Y_violent, Y_violent_pred_lasso)

# Log fit
beta_hat_v_log_lasso = lasso_coordinate_descent(X_final, Y_violent_log_naive, lmbda_lasso)
Y_violent_exp_pred_lasso = np.exp(X_final @ beta_hat_v_log_lasso)
mape_v_exp_lasso = mape(Y_violent, Y_violent_exp_pred_lasso)

print("\nLasso Regression Results (from scratch):")
print(f"Violent Crime MAPE (direct) lasso: {mape_v_lasso:.2f}")
print(f"Violent Crime MAPE (log fit on y) lasso: {mape_v_exp_lasso:.2f}")

# Non-violent
beta_hat_nv_lasso = lasso_coordinate_descent(X_final, Y_non_violent, lmbda_lasso)
Y_non_violent_pred_lasso = X_final @ beta_hat_nv_lasso
mape_nv_lasso = mape(Y_non_violent, Y_non_violent_pred_lasso)

# Log fit
beta_hat_nv_log_lasso = lasso_coordinate_descent(X_final, Y_non_violent_log_naive, lmbda_lasso)
Y_non_violent_exp_pred_lasso = np.exp(X_final @ beta_hat_nv_log_lasso)
mape_nv_exp_lasso = mape(Y_non_violent, Y_non_violent_exp_pred_lasso)

print(f"Non Violent Crime MAPE (direct) lasso: {mape_nv_lasso:.2f}")
print(f"Non Violent Crime MAPE (log fit on y) lasso: {mape_nv_exp_lasso:.2f}")

# Test set predictions with lasso
Y_violent_test_pred_lasso = X_test_final @ beta_hat_v_lasso
mape_v_test_lasso = mape(Y_violent_test, Y_violent_test_pred_lasso)
Y_violent_test_exp_pred_lasso = np.exp(X_test_final @ beta_hat_v_log_lasso)
mape_v_test_exp_lasso = mape(Y_violent_test, Y_violent_test_exp_pred_lasso)
print(f"Violent Crime Test MAPE (direct) lasso: {mape_v_test_lasso:.2f}")
print(f"Violent Crime Test MAPE (log fit on y) lasso: {mape_v_test_exp_lasso:.2f}")

Y_non_violent_test_pred_lasso = X_test_final @ beta_hat_nv_lasso
mape_nv_test_lasso = mape(Y_non_violent_test, Y_non_violent_test_pred_lasso)
Y_non_violent_test_exp_pred_lasso = np.exp(X_test_final @ beta_hat_nv_log_lasso)
mape_nv_test_exp_lasso = mape(Y_non_violent_test, Y_non_violent_test_exp_pred_lasso)
print(f"Non Violent Crime Test MAPE (direct) lasso: {mape_nv_test_lasso:.2f}")
print(f"Non Violent Crime Test MAPE (log fit on y) lasso: {mape_nv_test_exp_lasso:.2f}")


# Robust Regression: Least Absolute Deviations (LAD)
def lad_regression(X, y, num_iters=100, tol=1e-4):
    n, d = X.shape
    w = np.linalg.lstsq(X, y, rcond=None)[0]  # initialize with least squares
    for it in range(num_iters):
        y_pred = X @ w
        residuals = y - y_pred
        weights = 1.0 / (np.abs(residuals) + 1e-8)
        W = np.diag(weights)
        w_new = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
        if np.linalg.norm(w_new - w, ord=1) < tol:
            break
        w = w_new
    return w

# Violent
beta_hat_v_lad = lad_regression(X_final, Y_violent)
Y_violent_pred_lad = X_final @ beta_hat_v_lad
mape_v_lad = mape(Y_violent, Y_violent_pred_lad)

# Log fit
beta_hat_v_log_lad = lad_regression(X_final, Y_violent_log_naive)
Y_violent_exp_pred_lad = np.exp(X_final @ beta_hat_v_log_lad)
mape_v_exp_lad = mape(Y_violent, Y_violent_exp_pred_lad)

print("\nLAD (Robust) Regression Results:")
print(f"Violent Crime MAPE (direct) lad: {mape_v_lad:.2f}")
print(f"Violent Crime MAPE (log fit on y) lad: {mape_v_exp_lad:.2f}")

# Non-violent
beta_hat_nv_lad = lad_regression(X_final, Y_non_violent)
Y_non_violent_pred_lad = X_final @ beta_hat_nv_lad
mape_nv_lad = mape(Y_non_violent, Y_non_violent_pred_lad)

# Log fit
beta_hat_nv_log_lad = lad_regression(X_final, Y_non_violent_log_naive)
Y_non_violent_exp_pred_lad = np.exp(X_final @ beta_hat_nv_log_lad)
mape_nv_exp_lad = mape(Y_non_violent, Y_non_violent_exp_pred_lad)

print(f"Non Violent Crime MAPE (direct) lad: {mape_nv_lad:.2f}")
print(f"Non Violent Crime MAPE (log fit on y) lad: {mape_nv_exp_lad:.2f}")

# Test set predictions with LAD
Y_violent_test_pred_lad = X_test_final @ beta_hat_v_lad
mape_v_test_lad = mape(Y_violent_test, Y_violent_test_pred_lad)
Y_violent_test_exp_pred_lad = np.exp(X_test_final @ beta_hat_v_log_lad)
mape_v_test_exp_lad = mape(Y_violent_test, Y_violent_test_exp_pred_lad)
print(f"Violent Crime Test MAPE (direct) lad: {mape_v_test_lad:.2f}")
print(f"Violent Crime Test MAPE (log fit on y) lad: {mape_v_test_exp_lad:.2f}")

Y_non_violent_test_pred_lad = X_test_final @ beta_hat_nv_lad
mape_nv_test_lad = mape(Y_non_violent_test, Y_non_violent_test_pred_lad)
Y_non_violent_test_exp_pred_lad = np.exp(X_test_final @ beta_hat_nv_log_lad)
mape_nv_test_exp_lad = mape(Y_non_violent_test, Y_non_violent_test_exp_pred_lad)
print(f"Non Violent Crime Test MAPE (direct) lad: {mape_nv_test_lad:.2f}")
print(f"Non Violent Crime Test MAPE (log fit on y) lad: {mape_nv_test_exp_lad:.2f}")



#4th Degree Polynomial Regression
def make_poly_features(X, degree=4):
    # Only powers of each feature, no cross-terms
    X_poly = [np.ones(X.shape[0])]
    for d in range(1, degree+1):
        X_poly.append(X**d)
    return np.column_stack(X_poly)

# Remove intercept column for polynomial expansion
X_poly_base = X_final[:, 1:]
X_final_poly = make_poly_features(X_poly_base, degree=3)
X_test_final_poly = make_poly_features(X_test_final[:, 1:], degree=3)

# Violent (direct)
beta_hat_v_poly, *_ = np.linalg.lstsq(X_final_poly, Y_violent, rcond=None)
Y_violent_pred_poly = X_final_poly @ beta_hat_v_poly
mape_v_poly = mape(Y_violent, Y_violent_pred_poly)
Y_violent_test_pred_poly = X_test_final_poly @ beta_hat_v_poly
mape_v_test_poly = mape(Y_violent_test, Y_violent_test_pred_poly)

# Violent (log fit)
beta_hat_v_log_poly, *_ = np.linalg.lstsq(X_final_poly, Y_violent_log_naive, rcond=None)
Y_violent_exp_pred_poly = np.exp(X_final_poly @ beta_hat_v_log_poly)
mape_v_exp_poly = mape(Y_violent, Y_violent_exp_pred_poly)
Y_violent_test_exp_pred_poly = np.exp(X_test_final_poly @ beta_hat_v_log_poly)
mape_v_test_exp_poly = mape(Y_violent_test, Y_violent_test_exp_pred_poly)

# Non-violent (direct)
beta_hat_nv_poly, *_ = np.linalg.lstsq(X_final_poly, Y_non_violent, rcond=None)
Y_non_violent_pred_poly = X_final_poly @ beta_hat_nv_poly
mape_nv_poly = mape(Y_non_violent, Y_non_violent_pred_poly)
Y_non_violent_test_pred_poly = X_test_final_poly @ beta_hat_nv_poly
mape_nv_test_poly = mape(Y_non_violent_test, Y_non_violent_test_pred_poly)

# Non-violent (log fit)
beta_hat_nv_log_poly, *_ = np.linalg.lstsq(X_final_poly, Y_non_violent_log_naive, rcond=None)
Y_non_violent_exp_pred_poly = np.exp(X_final_poly @ beta_hat_nv_log_poly)
mape_nv_exp_poly = mape(Y_non_violent, Y_non_violent_exp_pred_poly)
Y_non_violent_test_exp_pred_poly = np.exp(X_test_final_poly @ beta_hat_nv_log_poly)
mape_nv_test_exp_poly = mape(Y_non_violent_test, Y_non_violent_test_exp_pred_poly)

print("\n3rd Degree Polynomial Regression Results (manual, no scikit-learn):")
print(f"Violent Crime MAPE (direct train): {mape_v_poly:.2f}")
print(f"Violent Crime MAPE (direct test): {mape_v_test_poly:.2f}")
print(f"Violent Crime MAPE (log fit train): {mape_v_exp_poly:.2f}")
print(f"Violent Crime MAPE (log fit test): {mape_v_test_exp_poly:.2f}")
print(f"Non Violent Crime MAPE (direct train): {mape_nv_poly:.2f}")
print(f"Non Violent Crime MAPE (direct test): {mape_nv_test_poly:.2f}")
print(f"Non Violent Crime MAPE (log fit train): {mape_nv_exp_poly:.2f}")
print(f"Non Violent Crime MAPE (log fit test): {mape_nv_test_exp_poly:.2f}")

# --- Lasso (L1) Regression on nth Degree Polynomial Features (manual, coordinate descent) ---
lmbda_poly_lasso = 0.1  # You can tune this value

# Violent (direct)
beta_hat_v_poly_lasso = lasso_coordinate_descent(X_final_poly, Y_violent, lmbda_poly_lasso)
Y_violent_pred_poly_lasso = X_final_poly @ beta_hat_v_poly_lasso
mape_v_poly_lasso = mape(Y_violent, Y_violent_pred_poly_lasso)
Y_violent_test_pred_poly_lasso = X_test_final_poly @ beta_hat_v_poly_lasso
mape_v_test_poly_lasso = mape(Y_violent_test, Y_violent_test_pred_poly_lasso)

# Violent (log fit)
beta_hat_v_log_poly_lasso = lasso_coordinate_descent(X_final_poly, Y_violent_log_naive, lmbda_poly_lasso)
Y_violent_exp_pred_poly_lasso = np.exp(X_final_poly @ beta_hat_v_log_poly_lasso)
mape_v_exp_poly_lasso = mape(Y_violent, Y_violent_exp_pred_poly_lasso)
Y_violent_test_exp_pred_poly_lasso = np.exp(X_test_final_poly @ beta_hat_v_log_poly_lasso)
mape_v_test_exp_poly_lasso = mape(Y_violent_test, Y_violent_test_exp_pred_poly_lasso)

# Non-violent (direct)
beta_hat_nv_poly_lasso = lasso_coordinate_descent(X_final_poly, Y_non_violent, lmbda_poly_lasso)
Y_non_violent_pred_poly_lasso = X_final_poly @ beta_hat_nv_poly_lasso
mape_nv_poly_lasso = mape(Y_non_violent, Y_non_violent_pred_poly_lasso)
Y_non_violent_test_pred_poly_lasso = X_test_final_poly @ beta_hat_nv_poly_lasso
mape_nv_test_poly_lasso = mape(Y_non_violent_test, Y_non_violent_test_pred_poly_lasso)

# Non-violent (log fit)
beta_hat_nv_log_poly_lasso = lasso_coordinate_descent(X_final_poly, Y_non_violent_log_naive, lmbda_poly_lasso)
Y_non_violent_exp_pred_poly_lasso = np.exp(X_final_poly @ beta_hat_nv_log_poly_lasso)
mape_nv_exp_poly_lasso = mape(Y_non_violent, Y_non_violent_exp_pred_poly_lasso)
Y_non_violent_test_exp_pred_poly_lasso = np.exp(X_test_final_poly @ beta_hat_nv_log_poly_lasso)
mape_nv_test_exp_poly_lasso = mape(Y_non_violent_test, Y_non_violent_test_exp_pred_poly_lasso)

print("\nLasso on 3rd Degree Polynomial Features (manual):")
print(f"Violent Crime MAPE (direct train): {mape_v_poly_lasso:.2f}")
print(f"Violent Crime MAPE (direct test): {mape_v_test_poly_lasso:.2f}")
print(f"Violent Crime MAPE (log fit train): {mape_v_exp_poly_lasso:.2f}")
print(f"Violent Crime MAPE (log fit test): {mape_v_test_exp_poly_lasso:.2f}")
print(f"Non Violent Crime MAPE (direct train): {mape_nv_poly_lasso:.2f}")
print(f"Non Violent Crime MAPE (direct test): {mape_nv_test_poly_lasso:.2f}")
print(f"Non Violent Crime MAPE (log fit train): {mape_nv_exp_poly_lasso:.2f}")
print(f"Non Violent Crime MAPE (log fit test): {mape_nv_test_exp_poly_lasso:.2f}")

# --- Stepwise Linear Regression (Forward Selection, manual) ---
def stepwise_forward_selection(X, y, max_features=None, tol=1e-4):
    n, d = X.shape
    selected = [0]  # Always include intercept
    remaining = list(range(1, d))
    prev_mse = np.inf
    if max_features is None:
        max_features = d
    while len(selected) < max_features:
        best_mse = np.inf
        best_j = None
        for j in remaining:
            cols = selected + [j]
            X_sub = X[:, cols]
            beta, *_ = np.linalg.lstsq(X_sub, y, rcond=None)
            y_pred = X_sub @ beta
            mse = np.mean((y - y_pred) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_j = j
        if best_j is not None and prev_mse - best_mse > tol:
            selected.append(best_j)
            remaining.remove(best_j)
            prev_mse = best_mse
        else:
            break
    return selected

# Violent
selected_v = stepwise_forward_selection(X_final, Y_violent, max_features=10)
X_v_step = X_final[:, selected_v]
beta_hat_v_step, *_ = np.linalg.lstsq(X_v_step, Y_violent, rcond=None)
Y_violent_pred_step = X_v_step @ beta_hat_v_step
mape_v_step = mape(Y_violent, Y_violent_pred_step)
X_v_test_step = X_test_final[:, selected_v]
Y_violent_test_pred_step = X_v_test_step @ beta_hat_v_step
mape_v_test_step = mape(Y_violent_test, Y_violent_test_pred_step)

# Non-violent
selected_nv = stepwise_forward_selection(X_final, Y_non_violent, max_features=10)
X_nv_step = X_final[:, selected_nv]
beta_hat_nv_step, *_ = np.linalg.lstsq(X_nv_step, Y_non_violent, rcond=None)
Y_non_violent_pred_step = X_nv_step @ beta_hat_nv_step
mape_nv_step = mape(Y_non_violent, Y_non_violent_pred_step)
X_nv_test_step = X_test_final[:, selected_nv]
Y_non_violent_test_pred_step = X_nv_test_step @ beta_hat_nv_step
mape_nv_test_step = mape(Y_non_violent_test, Y_non_violent_test_pred_step)

print("\nStepwise Linear Regression (Forward Selection) Results:")
print(f"Violent Crime: selected features = {selected_v}")
print(f"Violent Crime MAPE (train): {mape_v_step:.2f}")
print(f"Violent Crime MAPE (test): {mape_v_test_step:.2f}")
print(f"Non Violent Crime: selected features = {selected_nv}")
print(f"Non Violent Crime MAPE (train): {mape_nv_step:.2f}")
print(f"Non Violent Crime MAPE (test): {mape_nv_test_step:.2f}")

# --- Stepwise Linear Regression (Forward Selection) for log fits ---
# Violent (log fit)
selected_v_log = stepwise_forward_selection(X_final, Y_violent_log_naive, max_features=10)
X_v_step_log = X_final[:, selected_v_log]
beta_hat_v_step_log, *_ = np.linalg.lstsq(X_v_step_log, Y_violent_log_naive, rcond=None)
Y_violent_exp_pred_step = np.exp(X_v_step_log @ beta_hat_v_step_log)
mape_v_exp_step = mape(Y_violent, Y_violent_exp_pred_step)
X_v_test_step_log = X_test_final[:, selected_v_log]
Y_violent_test_exp_pred_step = np.exp(X_v_test_step_log @ beta_hat_v_step_log)
mape_v_test_exp_step = mape(Y_violent_test, Y_violent_test_exp_pred_step)

# Non-violent (log fit)
selected_nv_log = stepwise_forward_selection(X_final, Y_non_violent_log_naive, max_features=10)
X_nv_step_log = X_final[:, selected_nv_log]
beta_hat_nv_step_log, *_ = np.linalg.lstsq(X_nv_step_log, Y_non_violent_log_naive, rcond=None)
Y_non_violent_exp_pred_step = np.exp(X_nv_step_log @ beta_hat_nv_step_log)
mape_nv_exp_step = mape(Y_non_violent, Y_non_violent_exp_pred_step)
X_nv_test_step_log = X_test_final[:, selected_nv_log]
Y_non_violent_test_exp_pred_step = np.exp(X_nv_test_step_log @ beta_hat_nv_step_log)
mape_nv_test_exp_step = mape(Y_non_violent_test, Y_non_violent_test_exp_pred_step)

print("\nStepwise Linear Regression (Forward Selection, log fit) Results:")
print(f"Violent Crime (log fit): selected features = {selected_v_log}")
print(f"Violent Crime MAPE (log fit train): {mape_v_exp_step:.2f}")
print(f"Violent Crime MAPE (log fit test): {mape_v_test_exp_step:.2f}")
print(f"Non Violent Crime (log fit): selected features = {selected_nv_log}")
print(f"Non Violent Crime MAPE (log fit train): {mape_nv_exp_step:.2f}")
print(f"Non Violent Crime MAPE (log fit test): {mape_nv_test_exp_step:.2f}")

# --- LAD Regression on 3rd Degree Polynomial Features
# 3rd degree polynomial features (already defined as X_final_poly, X_test_final_poly)

# Violent (direct)
beta_hat_v_poly_lad = lad_regression(X_final_poly, Y_violent)
Y_violent_pred_poly_lad = X_final_poly @ beta_hat_v_poly_lad
mape_v_poly_lad = mape(Y_violent, Y_violent_pred_poly_lad)
Y_violent_test_pred_poly_lad = X_test_final_poly @ beta_hat_v_poly_lad
mape_v_test_poly_lad = mape(Y_violent_test, Y_violent_test_pred_poly_lad)

# Violent (log fit)
beta_hat_v_log_poly_lad = lad_regression(X_final_poly, Y_violent_log_naive)
Y_violent_exp_pred_poly_lad = np.exp(X_final_poly @ beta_hat_v_log_poly_lad)
mape_v_exp_poly_lad = mape(Y_violent, Y_violent_exp_pred_poly_lad)
Y_violent_test_exp_pred_poly_lad = np.exp(X_test_final_poly @ beta_hat_v_log_poly_lad)
mape_v_test_exp_poly_lad = mape(Y_violent_test, Y_violent_test_exp_pred_poly_lad)

# Non-violent (direct)
beta_hat_nv_poly_lad = lad_regression(X_final_poly, Y_non_violent)
Y_non_violent_pred_poly_lad = X_final_poly @ beta_hat_nv_poly_lad
mape_nv_poly_lad = mape(Y_non_violent, Y_non_violent_pred_poly_lad)
Y_non_violent_test_pred_poly_lad = X_test_final_poly @ beta_hat_nv_poly_lad
mape_nv_test_poly_lad = mape(Y_non_violent_test, Y_non_violent_test_pred_poly_lad)

# Non-violent (log fit)
beta_hat_nv_log_poly_lad = lad_regression(X_final_poly, Y_non_violent_log_naive)
Y_non_violent_exp_pred_poly_lad = np.exp(X_final_poly @ beta_hat_nv_log_poly_lad)
mape_nv_exp_poly_lad = mape(Y_non_violent, Y_non_violent_exp_pred_poly_lad)
Y_non_violent_test_exp_pred_poly_lad = np.exp(X_test_final_poly @ beta_hat_nv_log_poly_lad)
mape_nv_test_exp_poly_lad = mape(Y_non_violent_test, Y_non_violent_test_exp_pred_poly_lad)

print("\nLAD on 3rd Degree Polynomial Features (manual):")
print(f"Violent Crime MAPE (direct train): {mape_v_poly_lad:.2f}")
print(f"Violent Crime MAPE (direct test): {mape_v_test_poly_lad:.2f}")
print(f"Violent Crime MAPE (log fit train): {mape_v_exp_poly_lad:.2f}")
print(f"Violent Crime MAPE (log fit test): {mape_v_test_exp_poly_lad:.2f}")
print(f"Non Violent Crime MAPE (direct train): {mape_nv_poly_lad:.2f}")
print(f"Non Violent Crime MAPE (direct test): {mape_nv_test_poly_lad:.2f}")
print(f"Non Violent Crime MAPE (log fit train): {mape_nv_exp_poly_lad:.2f}")
print(f"Non Violent Crime MAPE (log fit test): {mape_nv_test_exp_poly_lad:.2f}")



# --- LAD with Ridge Regularization (Elastic LAD, manual IRLS+L2) ---
def lad_ridge_regression(X, y, lmbda=1.0, num_iters=100, tol=1e-4):
    n, d = X.shape
    w = np.linalg.lstsq(X, y, rcond=None)[0]  # initialize with least squares
    I = np.eye(d)
    I[0, 0] = 0  # do not regularize intercept
    for it in range(num_iters):
        y_pred = X @ w
        residuals = y - y_pred
        weights = 1.0 / (np.abs(residuals) + 1e-8)
        W = np.diag(weights)
        # Ridge-regularized weighted least squares
        A = W @ X
        b = W @ y
        w_new = np.linalg.inv(A.T @ A + lmbda * I) @ (A.T @ b)
        if np.linalg.norm(w_new - w, ord=1) < tol:
            break
        w = w_new
    return w

lmbda_lad_ridge = 1.0  # You can tune this value

# Violent (direct)
beta_hat_v_poly_lad_ridge = lad_ridge_regression(X_final_poly, Y_violent, lmbda=lmbda_lad_ridge)
Y_violent_pred_poly_lad_ridge = X_final_poly @ beta_hat_v_poly_lad_ridge
mape_v_poly_lad_ridge = mape(Y_violent, Y_violent_pred_poly_lad_ridge)
Y_violent_test_pred_poly_lad_ridge = X_test_final_poly @ beta_hat_v_poly_lad_ridge
mape_v_test_poly_lad_ridge = mape(Y_violent_test, Y_violent_test_pred_poly_lad_ridge)

# Violent (log fit)
beta_hat_v_log_poly_lad_ridge = lad_ridge_regression(X_final_poly, Y_violent_log_naive, lmbda=lmbda_lad_ridge)
Y_violent_exp_pred_poly_lad_ridge = np.exp(X_final_poly @ beta_hat_v_log_poly_lad_ridge)
mape_v_exp_poly_lad_ridge = mape(Y_violent, Y_violent_exp_pred_poly_lad_ridge)
Y_violent_test_exp_pred_poly_lad_ridge = np.exp(X_test_final_poly @ beta_hat_v_log_poly_lad_ridge)
mape_v_test_exp_poly_lad_ridge = mape(Y_violent_test, Y_violent_test_exp_pred_poly_lad_ridge)

# Non-violent (direct)
beta_hat_nv_poly_lad_ridge = lad_ridge_regression(X_final_poly, Y_non_violent, lmbda=lmbda_lad_ridge)
Y_non_violent_pred_poly_lad_ridge = X_final_poly @ beta_hat_nv_poly_lad_ridge
mape_nv_poly_lad_ridge = mape(Y_non_violent, Y_non_violent_pred_poly_lad_ridge)
Y_non_violent_test_pred_poly_lad_ridge = X_test_final_poly @ beta_hat_nv_poly_lad_ridge
mape_nv_test_poly_lad_ridge = mape(Y_non_violent_test, Y_non_violent_test_pred_poly_lad_ridge)

# Non-violent (log fit)
beta_hat_nv_log_poly_lad_ridge = lad_ridge_regression(X_final_poly, Y_non_violent_log_naive, lmbda=lmbda_lad_ridge)
Y_non_violent_exp_pred_poly_lad_ridge = np.exp(X_final_poly @ beta_hat_nv_log_poly_lad_ridge)
mape_nv_exp_poly_lad_ridge = mape(Y_non_violent, Y_non_violent_exp_pred_poly_lad_ridge)
Y_non_violent_test_exp_pred_poly_lad_ridge = np.exp(X_test_final_poly @ beta_hat_nv_log_poly_lad_ridge)
mape_nv_test_exp_poly_lad_ridge = mape(Y_non_violent_test, Y_non_violent_test_exp_pred_poly_lad_ridge)

print("\nLAD with Ridge Regularization on 3rd Degree Polynomial Features (manual):")
print(f"Violent Crime MAPE (direct train): {mape_v_poly_lad_ridge:.2f}")
print(f"Violent Crime MAPE (direct test): {mape_v_test_poly_lad_ridge:.2f}")
print(f"Violent Crime MAPE (log fit train): {mape_v_exp_poly_lad_ridge:.2f}")
print(f"Violent Crime MAPE (log fit test): {mape_v_test_exp_poly_lad_ridge:.2f}")
print(f"Non Violent Crime MAPE (direct train): {mape_nv_poly_lad_ridge:.2f}")
print(f"Non Violent Crime MAPE (direct test): {mape_nv_test_poly_lad_ridge:.2f}")
print(f"Non Violent Crime MAPE (log fit train): {mape_nv_exp_poly_lad_ridge:.2f}")
print(f"Non Violent Crime MAPE (log fit test): {mape_nv_test_exp_poly_lad_ridge:.2f}")
lmbda_lad_ridge_linear = 1.0  # You can tune this value

# Violent (direct)
beta_hat_v_lad_ridge = lad_ridge_regression(X_final, Y_violent, lmbda=lmbda_lad_ridge_linear)
Y_violent_pred_lad_ridge = X_final @ beta_hat_v_lad_ridge
mape_v_lad_ridge = mape(Y_violent, Y_violent_pred_lad_ridge)
Y_violent_test_pred_lad_ridge = X_test_final @ beta_hat_v_lad_ridge
mape_v_test_lad_ridge = mape(Y_violent_test, Y_violent_test_pred_lad_ridge)

# Violent (log fit)
beta_hat_v_log_lad_ridge = lad_ridge_regression(X_final, Y_violent_log_naive, lmbda=lmbda_lad_ridge_linear)
Y_violent_exp_pred_lad_ridge = np.exp(X_final @ beta_hat_v_log_lad_ridge)
mape_v_exp_lad_ridge = mape(Y_violent, Y_violent_exp_pred_lad_ridge)
Y_violent_test_exp_pred_lad_ridge = np.exp(X_test_final @ beta_hat_v_log_lad_ridge)
mape_v_test_exp_lad_ridge = mape(Y_violent_test, Y_violent_test_exp_pred_lad_ridge)

# Non-violent (direct)
beta_hat_nv_lad_ridge = lad_ridge_regression(X_final, Y_non_violent, lmbda=lmbda_lad_ridge_linear)
Y_non_violent_pred_lad_ridge = X_final @ beta_hat_nv_lad_ridge
mape_nv_lad_ridge = mape(Y_non_violent, Y_non_violent_pred_lad_ridge)
Y_non_violent_test_pred_lad_ridge = X_test_final @ beta_hat_nv_lad_ridge
mape_nv_test_lad_ridge = mape(Y_non_violent_test, Y_non_violent_test_pred_lad_ridge)

# Non-violent (log fit)
beta_hat_nv_log_lad_ridge = lad_ridge_regression(X_final, Y_non_violent_log_naive, lmbda=lmbda_lad_ridge_linear)
Y_non_violent_exp_pred_lad_ridge = np.exp(X_final @ beta_hat_nv_log_lad_ridge)
mape_nv_exp_lad_ridge = mape(Y_non_violent, Y_non_violent_exp_pred_lad_ridge)
Y_non_violent_test_exp_pred_lad_ridge = np.exp(X_test_final @ beta_hat_nv_log_lad_ridge)
mape_nv_test_exp_lad_ridge = mape(Y_non_violent_test, Y_non_violent_test_exp_pred_lad_ridge)

print("\nLAD with Ridge Regularization on Linear Features (manual):")
print(f"Violent Crime MAPE (direct train): {mape_v_lad_ridge:.2f}")
print(f"Violent Crime MAPE (direct test): {mape_v_test_lad_ridge:.2f}")
print(f"Violent Crime MAPE (log fit train): {mape_v_exp_lad_ridge:.2f}")
print(f"Violent Crime MAPE (log fit test): {mape_v_test_exp_lad_ridge:.2f}")
print(f"Non Violent Crime MAPE (direct train): {mape_nv_lad_ridge:.2f}")
print(f"Non Violent Crime MAPE (direct test): {mape_nv_test_lad_ridge:.2f}")
print(f"Non Violent Crime MAPE (log fit train): {mape_nv_exp_lad_ridge:.2f}")
print(f"Non Violent Crime MAPE (log fit test): {mape_nv_test_exp_lad_ridge:.2f}")

