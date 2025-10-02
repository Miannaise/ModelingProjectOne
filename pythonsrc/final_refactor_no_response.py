import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import makedirs, path
import scipy.stats as stats

# --- Preprocessing ---
def normalize_feature(x, y):
    mean = np.mean(y, axis=0, keepdims=True)
    std = np.std(y, axis=0, keepdims=True)
    return (x - mean) / (std + 1e-8)

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

def compute_kurtosis(X):
    n = X.shape[0]
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return np.sum(((X - mean) / (std + 1e-8))**4, axis=0) / n - 3

def drop_high_kurtosis_features(X, kurtosis_vector, threshold=10.0):
    to_drop = [i for i, k in enumerate(kurtosis_vector) if abs(k) > threshold]
    keep_indices = [i for i in range(X.shape[1]) if i not in to_drop]
    return X[:, keep_indices], keep_indices

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# load mat, also it's weird and a dict for some reason? maybe that's how .mat is.
training_data = scio.loadmat("./proj1files/trainingdata.mat")["trainingdata"]
X_train = training_data[:, 0:78]
Y_violent = training_data[:, 79]
Y_non_violent = training_data[:, 80]

normalize_features = normalize_feature(X_train, X_train)
covar = np.cov(normalize_features, rowvar=False)

X_dropped_normalized, kept_indices = dropout_highly_correlated_features(normalize_features, threshold=0.9)
kurtosis_vector = compute_kurtosis(X_dropped_normalized)
X_final, final_kept_indices = drop_high_kurtosis_features(X_dropped_normalized, kurtosis_vector, threshold=10)
X_final = np.insert(X_final, 0, np.ones((X_final.shape[0],)), axis=1)  # add intercept

#Test Data
testing_data = scio.loadmat("./proj1files/testingdata.mat")["testingdata"]
X_test = normalize_feature(testing_data[:, 0:78], X_train)
Y_violent_test = testing_data[:, 79]
Y_non_violent_test = testing_data[:, 80]
X_test_dropped = X_test[:, kept_indices]
X_test_final = X_test_dropped[:, final_kept_indices]
X_test_final = np.insert(X_test_final, 0, np.ones((X_test_final.shape[0],)), axis=1)

#Regression Utilities
def ridge_regression(X, y, lmbda):
    n_features = X.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0
    return np.linalg.inv(X.T @ X + lmbda * I) @ X.T @ y

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
    for _ in range(num_iters):
        w_old = w.copy()
        for j in range(n_features):
            X_j = X[:, j]
            y_pred = X @ w
            residual = y - y_pred + w[j] * X_j
            rho = X_j @ residual
            denom = X_j @ X_j
            if np.abs(denom) < 1e-12:
                w[j] = 0.0
                continue
            if j == 0:
                w[j] = rho / denom
            else:
                w[j] = soft_thresholding(rho, lmbda / 2) / denom
        if np.linalg.norm(w - w_old, ord=1) < tol:
            break
    return w

def lad_regression(X, y, num_iters=100, tol=1e-4):
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(num_iters):
        y_pred = X @ w
        residuals = y - y_pred
        weights = 1.0 / (np.abs(residuals) + 1e-8)
        W = np.diag(weights)
        w_new = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
        if np.linalg.norm(w_new - w, ord=1) < tol:
            break
        w = w_new
    return w

def lad_ridge_regression(X, y, lmbda=1e-4, num_iters=100, tol=1e-4):
    n, d = X.shape
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    I = np.eye(d)
    I[0, 0] = 0
    for _ in range(num_iters):
        y_pred = X @ w
        residuals = y - y_pred
        weights = 1.0 / (np.abs(residuals) + 1e-8)
        W = np.diag(weights)
        A = W @ X
        b = W @ y
        w_new = np.linalg.inv(A.T @ A + lmbda * I) @ (A.T @ b)
        if np.linalg.norm(w_new - w, ord=1) < tol:
            break
        w = w_new
    return w

def make_poly_features(X, degree=3):
    X_poly = [np.ones(X.shape[0])]
    for d in range(1, degree + 1):
        X_poly.append(X ** d)
    return np.column_stack(X_poly)

def ridge_regression_safe(X, y, lmbda):
    n_features = X.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0
    reg_matrix = lmbda * I + 1e-8 * np.eye(n_features)
    A = X.T @ X + reg_matrix
    return np.linalg.pinv(A) @ (X.T @ y)
#Stepwise Linear Regression (Forward Selection)
def stepwise_forward_selection(X, y, max_features=None, tol=1e-4):
    n, d = X.shape
    selected = [0]  # Always include intercept
    remaining = list(range(1, d))
    prev_mape = np.inf
    if max_features is None:
        max_features = d
    while len(selected) < max_features:
        best_mape = np.inf
        best_j = None
        for j in remaining:
            cols = selected + [j]
            X_sub = X[:, cols]
            beta, *_ = np.linalg.lstsq(X_sub, y, rcond=None)
            y_pred = X_sub @ beta
            mape_value = mape(y, y_pred)
            if mape_value < best_mape:
                best_mape = mape_value
                best_j = j
        if best_j is not None and prev_mape - best_mape > tol:
            selected.append(best_j)
            remaining.remove(best_j)
            prev_mape = best_mape
        else:
            break
    return selected

#polynomial and SVD
X_poly_base = X_final[:, 1:]
X_final_poly = make_poly_features(X_poly_base, degree=3)
X_test_final_poly = make_poly_features(X_test_final[:, 1:], degree=3)

U, S, VT = np.linalg.svd(X_final, full_matrices=False)
X_svd = U @ np.diag(S)
X_test_svd = X_test_final @ VT.T

U_poly, S_poly, VT_poly = np.linalg.svd(X_final_poly, full_matrices=False)
X_svd_poly = U_poly @ np.diag(S_poly)
X_test_svd_poly = X_test_final_poly @ VT_poly.T

#Log Transform
Y_violent_log = np.log(np.clip(Y_violent, 1e-5, None))
Y_non_violent_log = np.log(np.clip(Y_non_violent, 1e-5, None))
Y_violent_test_log = np.log(np.clip(Y_violent_test, 1e-5, None))
Y_non_violent_test_log = np.log(np.clip(Y_non_violent_test, 1e-5, None))

# Modeling and Evaluation
def evaluate_model(X_train, y_train, X_test, y_test, fit_func, fit_args=(), log_fit=False):
    if log_fit:
        y_train_fit = np.log(np.clip(y_train, 1e-5, None))
        beta = fit_func(X_train, y_train_fit, *fit_args)
        if isinstance(beta, tuple):  # handle np.linalg.lstsq
            beta = beta[0]
        y_pred_train = np.exp(X_train @ beta)
        y_pred_test = np.exp(X_test @ beta)
    else:
        beta = fit_func(X_train, y_train, *fit_args)
        if isinstance(beta, tuple):  # handle np.linalg.lstsq
            beta = beta[0]
        y_pred_train = X_train @ beta
        y_pred_test = X_test @ beta
    return mape(y_train, y_pred_train), mape(y_test, y_pred_test), beta, y_pred_train, y_pred_test

def print_results(name, mape_train, mape_test, log_fit=False):
    prefix = "Log " if log_fit else ""
    print(f"{name} {prefix}MAPE (train): {mape_train:.2f}")
    print(f"{name} {prefix}MAPE (test): {mape_test:.2f}")

print(f"Initial shape: {X_train.shape}")
print(f"Shape after dropping highly correlated features: {X_dropped_normalized.shape}")
print(f"Final shape after dropping high kurtosis features (with intercept): {X_final.shape}")
#Run the models
lmbda = 3e-7

# 1) Linear
mape_v, mape_vt, *_ = evaluate_model(X_final, Y_violent, X_test_final, Y_violent_test, np.linalg.lstsq)
mape_nv, mape_nvt, *_ = evaluate_model(X_final, Y_non_violent, X_test_final, Y_non_violent_test, np.linalg.lstsq)
print_results("Linear", mape_v, mape_vt)
print_results("Linear", mape_nv, mape_nvt)

# 2) Linear Log
mape_vl, mape_vtl, *_ = evaluate_model(X_final, Y_violent, X_test_final, Y_violent_test, np.linalg.lstsq, log_fit=True)
mape_nvl, mape_nvtl, *_ = evaluate_model(X_final, Y_non_violent, X_test_final, Y_non_violent_test, np.linalg.lstsq, log_fit=True)
print_results("Linear", mape_vl, mape_vtl, log_fit=True)
print_results("Linear", mape_nvl, mape_nvtl, log_fit=True)

# 3) Ridge
mape_v, mape_vt, *_ = evaluate_model(X_final, Y_violent, X_test_final, Y_violent_test, ridge_regression, (lmbda,))
mape_nv, mape_nvt, *_ = evaluate_model(X_final, Y_non_violent, X_test_final, Y_non_violent_test, ridge_regression, (lmbda,))
print_results("Ridge", mape_v, mape_vt)
print_results("Ridge", mape_nv, mape_nvt)

# 4) Ridge Log
mape_vl, mape_vtl, *_ = evaluate_model(X_final, Y_violent, X_test_final, Y_violent_test, ridge_regression, (lmbda,), log_fit=True)
mape_nvl, mape_nvtl, *_ = evaluate_model(X_final, Y_non_violent, X_test_final, Y_non_violent_test, ridge_regression, (lmbda,), log_fit=True)
print_results("Ridge", mape_vl, mape_vtl, log_fit=True)
print_results("Ridge", mape_nvl, mape_nvtl, log_fit=True)

# 5) Lasso
mape_v, mape_vt, *_ = evaluate_model(X_final, Y_violent, X_test_final, Y_violent_test, lasso_coordinate_descent, (lmbda,))
mape_nv, mape_nvt, *_ = evaluate_model(X_final, Y_non_violent, X_test_final, Y_non_violent_test, lasso_coordinate_descent, (lmbda,))
print_results("Lasso", mape_v, mape_vt)
print_results("Lasso", mape_nv, mape_nvt)

# 6) Lasso Log
mape_vl, mape_vtl, *_ = evaluate_model(X_final, Y_violent, X_test_final, Y_violent_test, lasso_coordinate_descent, (lmbda,), log_fit=True)
mape_nvl, mape_nvtl, *_ = evaluate_model(X_final, Y_non_violent, X_test_final, Y_non_violent_test, lasso_coordinate_descent, (lmbda,), log_fit=True)
print_results("Lasso", mape_vl, mape_vtl, log_fit=True)
print_results("Lasso", mape_nvl, mape_nvtl, log_fit=True)

# 7) LAD
mape_v, mape_vt, *_ = evaluate_model(X_final, Y_violent, X_test_final, Y_violent_test, lad_regression)
mape_nv, mape_nvt, *_ = evaluate_model(X_final, Y_non_violent, X_test_final, Y_non_violent_test, lad_regression)
print_results("LAD", mape_v, mape_vt)
print_results("LAD", mape_nv, mape_nvt)

# 8) LAD Log
mape_vl, mape_vtl, *_ = evaluate_model(X_final, Y_violent, X_test_final, Y_violent_test, lad_regression, (), log_fit=True)
mape_nvl, mape_nvtl, *_ = evaluate_model(X_final, Y_non_violent, X_test_final, Y_non_violent_test, lad_regression, (), log_fit=True)
print_results("LAD", mape_vl, mape_vtl, log_fit=True)
print_results("LAD", mape_nvl, mape_nvtl, log_fit=True)

# 9) Polynomial
mape_v, mape_vt, *_ = evaluate_model(X_final_poly, Y_violent, X_test_final_poly, Y_violent_test, np.linalg.lstsq)
mape_nv, mape_nvt, *_ = evaluate_model(X_final_poly, Y_non_violent, X_test_final_poly, Y_non_violent_test, np.linalg.lstsq)
print_results("Poly3", mape_v, mape_vt)
print_results("Poly3", mape_nv, mape_nvt)

# 10) Polynomial Log
mape_vl, mape_vtl, *_ = evaluate_model(X_final_poly, Y_violent, X_test_final_poly, Y_violent_test, np.linalg.lstsq, log_fit=True)
mape_nvl, mape_nvtl, *_ = evaluate_model(X_final_poly, Y_non_violent, X_test_final_poly, Y_non_violent_test, np.linalg.lstsq, log_fit=True)
print_results("Poly3", mape_vl, mape_vtl, log_fit=True)
print_results("Poly3", mape_nvl, mape_nvtl, log_fit=True)

# 11) Poly Lasso
mape_v, mape_vt, *_ = evaluate_model(X_final_poly, Y_violent, X_test_final_poly, Y_violent_test, lasso_coordinate_descent, (lmbda,))
mape_nv, mape_nvt, *_ = evaluate_model(X_final_poly, Y_non_violent, X_test_final_poly, Y_non_violent_test, lasso_coordinate_descent, (lmbda,))
print_results("Poly3 Lasso", mape_v, mape_vt)
print_results("Poly3 Lasso", mape_nv, mape_nvt)

# 12) Poly Lasso Log
mape_vl, mape_vtl, *_ = evaluate_model(X_final_poly, Y_violent, X_test_final_poly, Y_violent_test, lasso_coordinate_descent, (lmbda,), log_fit=True)
mape_nvl, mape_nvtl, *_ = evaluate_model(X_final_poly, Y_non_violent, X_test_final_poly, Y_non_violent_test, lasso_coordinate_descent, (lmbda,), log_fit=True)
print_results("Poly3 Lasso", mape_vl, mape_vtl, log_fit=True)
print_results("Poly3 Lasso", mape_nvl, mape_nvtl, log_fit=True)

# 13) Poly LAD
mape_v, mape_vt, *_ = evaluate_model(X_final_poly, Y_violent, X_test_final_poly, Y_violent_test, lad_regression)
mape_nv, mape_nvt, *_ = evaluate_model(X_final_poly, Y_non_violent, X_test_final_poly, Y_non_violent_test, lad_regression)
print_results("Poly3 LAD", mape_v, mape_vt)
print_results("Poly3 LAD", mape_nv, mape_nvt)

# 14) Poly LAD Log
mape_vl, mape_vtl, *_ = evaluate_model(X_final_poly, Y_violent, X_test_final_poly, Y_violent_test, lad_regression, (), log_fit=True)
mape_nvl, mape_nvtl, *_ = evaluate_model(X_final_poly, Y_non_violent, X_test_final_poly, Y_non_violent_test, lad_regression, (), log_fit=True)
print_results("Poly3 LAD", mape_vl, mape_vtl, log_fit=True)
print_results("Poly3 LAD", mape_nvl, mape_nvtl, log_fit=True)

# 15) Poly LAD Ridge
mape_v, mape_vt, *_ = evaluate_model(X_final_poly, Y_violent, X_test_final_poly, Y_violent_test, lad_ridge_regression, (lmbda,))
mape_nv, mape_nvt, *_ = evaluate_model(X_final_poly, Y_non_violent, X_test_final_poly, Y_non_violent_test, lad_ridge_regression, (lmbda,))
print_results("Poly3 LAD Ridge", mape_v, mape_vt)
print_results("Poly3 LAD Ridge", mape_nv, mape_nvt)

# 16) Poly LAD Ridge Log
mape_vl, mape_vtl, *_ = evaluate_model(X_final_poly, Y_violent, X_test_final_poly, Y_violent_test, lad_ridge_regression, (lmbda,), log_fit=True)
mape_nvl, mape_nvtl, *_ = evaluate_model(X_final_poly, Y_non_violent, X_test_final_poly, Y_non_violent_test, lad_ridge_regression, (lmbda,), log_fit=True)
print_results("Poly3 LAD Ridge", mape_vl, mape_vtl, log_fit=True)
print_results("Poly3 LAD Ridge", mape_nvl, mape_nvtl, log_fit=True)

# 17) SVD Ridge
mape_v, mape_vt, *_ = evaluate_model(X_svd, Y_violent, X_test_svd, Y_violent_test, ridge_regression_safe, (lmbda,))
mape_nv, mape_nvt, *_ = evaluate_model(X_svd, Y_non_violent, X_test_svd, Y_non_violent_test, ridge_regression_safe, (lmbda,))
print_results("SVD Ridge", mape_v, mape_vt)
print_results("SVD Ridge", mape_nv, mape_nvt)

# 18) SVD Ridge Log
mape_vl, mape_vtl, *_ = evaluate_model(X_svd, Y_violent, X_test_svd, Y_violent_test, ridge_regression_safe, (lmbda,), log_fit=True)
mape_nvl, mape_nvtl, *_ = evaluate_model(X_svd, Y_non_violent, X_test_svd, Y_non_violent_test, ridge_regression_safe, (lmbda,), log_fit=True)
print_results("SVD Ridge", mape_vl, mape_vtl, log_fit=True)
print_results("SVD Ridge", mape_nvl, mape_nvtl, log_fit=True)

# 19) SVD Poly Ridge
mape_v, mape_vt, *_ = evaluate_model(X_svd_poly, Y_violent, X_test_svd_poly, Y_violent_test, ridge_regression_safe, (lmbda,))
mape_nv, mape_nvt, *_ = evaluate_model(X_svd_poly, Y_non_violent, X_test_svd_poly, Y_non_violent_test, ridge_regression_safe, (lmbda,))
print_results("SVD Poly Ridge", mape_v, mape_vt)
print_results("SVD Poly Ridge", mape_nv, mape_nvt)

# 20) SVD Poly Ridge Log
mape_vl, mape_vtl, *_ = evaluate_model(X_svd_poly, Y_violent, X_test_svd_poly, Y_violent_test, ridge_regression_safe, (lmbda,), log_fit=True)
mape_nvl, mape_nvtl, *_ = evaluate_model(X_svd_poly, Y_non_violent, X_test_svd_poly, Y_non_violent_test, ridge_regression_safe, (lmbda,), log_fit=True)
print_results("SVD Poly Ridge", mape_vl, mape_vtl, log_fit=True)
print_results("SVD Poly Ridge", mape_nvl, mape_nvtl, log_fit=True)


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

#Stepwise Linear Regression (Forward Selection) for log fits
# Violent (log fit)
selected_v_log = stepwise_forward_selection(X_final, Y_violent_log, max_features=10)
X_v_step_log = X_final[:, selected_v_log]
beta_hat_v_step_log, *_ = np.linalg.lstsq(X_v_step_log, Y_violent_log, rcond=None)
Y_violent_exp_pred_step = np.exp(X_v_step_log @ beta_hat_v_step_log)
mape_v_exp_step = mape(Y_violent, Y_violent_exp_pred_step)
X_v_test_step_log = X_test_final[:, selected_v_log]
Y_violent_test_exp_pred_step = np.exp(X_v_test_step_log @ beta_hat_v_step_log)
mape_v_test_exp_step = mape(Y_violent_test, Y_violent_test_exp_pred_step)

# Non-violent (log fit)
selected_nv_log = stepwise_forward_selection(X_final, Y_non_violent_log, max_features=10)
X_nv_step_log = X_final[:, selected_nv_log]
beta_hat_nv_step_log, *_ = np.linalg.lstsq(X_nv_step_log, Y_non_violent_log, rcond=None)
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

#Perform stepwise forward selection with all normalized features
normalize_all_features = normalize_feature(X_train, X_train)
selected_all_features_v = stepwise_forward_selection(normalize_all_features, Y_violent, max_features=normalize_all_features.shape[1])
selected_all_features_nv = stepwise_forward_selection(normalize_all_features, Y_non_violent, max_features=normalize_all_features.shape[1])

# Violent Crime
X_v_all_step = normalize_all_features[:, selected_all_features_v]
beta_hat_v_all_step, *_ = np.linalg.lstsq(X_v_all_step, Y_violent, rcond=None)
Y_violent_pred_all_step = X_v_all_step @ beta_hat_v_all_step
mape_v_all_step = mape(Y_violent, Y_violent_pred_all_step)

# Non-Violent Crime
X_nv_all_step = normalize_all_features[:, selected_all_features_nv]
beta_hat_nv_all_step, *_ = np.linalg.lstsq(X_nv_all_step, Y_non_violent, rcond=None)
Y_non_violent_pred_all_step = X_nv_all_step @ beta_hat_nv_all_step
mape_nv_all_step = mape(Y_non_violent, Y_non_violent_pred_all_step)

print("\nStepwise Linear Regression (All Features, Normalized) Results:")
print(f"Violent Crime: selected features = {selected_all_features_v}")
print(f"Violent Crime MAPE (train): {mape_v_all_step:.2f}")
print(f"Non Violent Crime: selected features = {selected_all_features_nv}")
print(f"Non Violent Crime MAPE (train): {mape_nv_all_step:.2f}")

# Perform stepwise forward selection with all normalized features (log fit)
selected_all_features_v_log = stepwise_forward_selection(normalize_all_features, Y_violent_log, max_features=normalize_all_features.shape[1])
selected_all_features_nv_log = stepwise_forward_selection(normalize_all_features, Y_non_violent_log, max_features=normalize_all_features.shape[1])

# Violent Crime (log fit)
X_v_all_step_log = normalize_all_features[:, selected_all_features_v_log]
beta_hat_v_all_step_log, *_ = np.linalg.lstsq(X_v_all_step_log, Y_violent_log, rcond=None)
Y_violent_pred_all_step_log = np.exp(X_v_all_step_log @ beta_hat_v_all_step_log)
mape_v_all_step_log = mape(Y_violent, Y_violent_pred_all_step_log)

# Non-Violent Crime (log fit)
X_nv_all_step_log = normalize_all_features[:, selected_all_features_nv_log]
beta_hat_nv_all_step_log, *_ = np.linalg.lstsq(X_nv_all_step_log, Y_non_violent_log, rcond=None)
Y_non_violent_pred_all_step_log = np.exp(X_nv_all_step_log @ beta_hat_nv_all_step_log)
mape_nv_all_step_log = mape(Y_non_violent, Y_non_violent_pred_all_step_log)

print("\nStepwise Linear Regression (All Features, Normalized, Log Fit) Results:")
print(f"Violent Crime (log fit): selected features = {selected_all_features_v_log}")
print(f"Violent Crime MAPE (log fit train): {mape_v_all_step_log:.2f}")
print(f"Non Violent Crime (log fit): selected features = {selected_all_features_nv_log}")
print(f"Non Violent Crime MAPE (log fit train): {mape_nv_all_step_log:.2f}")

#Plot the errors
def plot_errors(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True)
    plt.show()
    plt.close()
# Scatter plots of percent error for Naiive (Linear) and Poly3 Lasso Log models
def scatter_percent_error(y_true, y_pred, title):
    percent_error = 100 * (y_true - y_pred) / (y_true + 1e-8)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=percent_error)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Percent Error (%)')
    plt.title(title)
    plt.grid(True)
    plt.show()
    plt.close()

# Example plots
plot_errors(Y_violent, X_final @ np.linalg.lstsq(X_final, Y_violent, rcond=None)[0], "Linear Regression Violent Crime")
plot_errors(Y_non_violent, X_final @ np.linalg.lstsq(X_final, Y_non_violent, rcond=None)[0], "Linear Regression Non-Violent Crime")
plot_errors(Y_violent, np.exp(X_final @ np.linalg.lstsq(X_final, Y_violent_log, rcond=None)[0]), "Log Linear Regression Violent Crime")
plot_errors(Y_non_violent, np.exp(X_final @ np.linalg.lstsq(X_final, Y_non_violent_log, rcond=None)[0]), "Log Linear Regression Non-Violent Crime")


#Histograms of errors
def plot_error_histogram(y_true, y_pred, title):
    errors = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, kde=True, bins=30, color='blue')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(f"Error Histogram: {title}")
    plt.grid(True)
    plt.show()
    plt.close()

# Example histograms
plot_error_histogram(Y_violent, X_final @ np.linalg.lstsq(X_final, Y_violent, rcond=None)[0], "Linear Regression Violent Crime")
plot_error_histogram(Y_non_violent, X_final @ np.linalg.lstsq(X_final, Y_non_violent, rcond=None)[0], "Linear Regression Non-Violent Crime")
plot_error_histogram(Y_violent, np.exp(X_final @ np.linalg.lstsq(X_final, Y_violent_log, rcond=None)[0]), "Log Linear Regression Violent Crime")
plot_error_histogram(Y_non_violent, np.exp(X_final @ np.linalg.lstsq(X_final, Y_non_violent_log, rcond=None)[0]), "Log Linear Regression Non-Violent Crime")


#Q-Q plots of errors
def plot_qq_errors(y_true, y_pred, title):
    errors = y_true - y_pred
    plt.figure(figsize=(8, 6))
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot: {title}")
    plt.grid(True)
    plt.show()
    plt.close()



def scatter_error(y_true, y_pred, title):
    errors = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=errors)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Error (True - Predicted)')
    plt.title(title)
    plt.grid(True)
    plt.show()
    plt.close()


# Naiive (Linear) model
beta_linear = np.linalg.lstsq(X_final, Y_violent, rcond=None)[0]
y_pred_linear = X_final @ beta_linear
scatter_error(Y_violent, y_pred_linear, "Error Scatter: Naiive Linear Violent Crime")
scatter_percent_error(Y_violent, y_pred_linear, "Percent Error Scatter: Naiive Linear Violent Crime")
#log model
beta_log = np.linalg.lstsq(X_final, Y_violent_log, rcond=None)[0]
y_pred_log = np.exp(X_final @ beta_log)
scatter_error(Y_violent, y_pred_log, "Error Scatter: Log Linear Violent Crime")
scatter_percent_error(Y_violent, y_pred_log, "Percent Error Scatter: Log Linear Violent Crime")
#non-violent log
beta_nv_log = np.linalg.lstsq(X_final, Y_non_violent_log, rcond=None)[0]
y_pred_nv_log = np.exp(X_final @ beta_nv_log)
scatter_error(Y_non_violent, y_pred_nv_log, "Error Scatter: Log Linear Non-Violent Crime")
scatter_percent_error(Y_non_violent, y_pred_nv_log, "Percent Error Scatter: Log Linear Non-Violent Crime")
# Poly3 Lasso Log model
beta_poly3_lasso_log = lasso_coordinate_descent(X_final_poly, np.log(np.clip(Y_violent, 1e-5, None)), lmbda)
y_pred_poly3_lasso_log = np.exp(X_final_poly @ beta_poly3_lasso_log)
scatter_error(Y_violent, y_pred_poly3_lasso_log, "Error Scatter: Poly3 Lasso Log Violent Crime")
scatter_percent_error(Y_violent, y_pred_poly3_lasso_log, "Percent Error Scatter: Poly3 Lasso Log Violent Crime")


#Q-Q plots for the best models
plot_qq_errors(Y_violent, X_final @ np.linalg.lstsq(X_final, Y_violent, rcond=None)[0], "Linear Regression Violent Crime")
plot_qq_errors(Y_non_violent, X_final @ np.linalg.lstsq(X_final, Y_non_violent, rcond=None)[0], "Linear Regression Non-Violent Crime")
plot_qq_errors(Y_violent, np.exp(X_final @ np.linalg.lstsq(X_final, Y_violent_log, rcond=None)[0]), "Log Linear Regression Violent Crime")
plot_qq_errors(Y_non_violent, np.exp(X_final @ np.linalg.lstsq(X_final, Y_non_violent_log, rcond=None)[0]), "Log Linear Regression Non-Violent Crime")
