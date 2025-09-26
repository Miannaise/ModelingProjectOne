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

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def fit_linear(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda x_: coeffs[0] * x_ + coeffs[1]

def fit_exponential(x, y):
    y = np.clip(y, 1e-5, None)
    log_y = np.log(y)
    A = np.vstack([x, np.ones_like(x)]).T
    coeffs = np.linalg.lstsq(A, log_y, rcond=None)[0]
    return lambda x_: np.exp(coeffs[0] * x_ + coeffs[1])

def fit_reciprocal(x, y):
    x = np.clip(x, 1e-5, None)
    A = np.vstack([1 / x, np.ones_like(x)]).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda x_: coeffs[0] / np.clip(x_, 1e-5, None) + coeffs[1]

def fit_logarithmic(x, y):
    x = np.clip(x, 1e-5, None)
    A = np.vstack([np.log(x), np.ones_like(x)]).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda x_: coeffs[0] * np.log(np.clip(x_, 1e-5, None)) + coeffs[1]

def fit_polynomial(x, y, degree):
    A = np.vstack([x**d for d in range(degree, -1, -1)]).T
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda x_: sum(c * x_**d for c, d in zip(coeffs, range(degree, -1, -1)))


def evaluate_fits(X_train, y, degree=4):
    results = []

    for i in range(X_train.shape[1]):
        x = X_train[:, i]

        # # Skip constant or invalid features
        # if np.all(x == x[0]):
        #     continue

        # Try each fit
       
        # Linear
        model = fit_linear(x, y)
        r2 = r_squared(y, model(x))
        results.append((i, "linear", r2))

        # Exponential
        model = fit_exponential(x, y)
        r2 = r_squared(y, model(x))
        results.append((i, "exponential", r2))

        # Reciprocal
        model = fit_reciprocal(x, y)
        r2 = r_squared(y, model(x))
        results.append((i, "reciprocal", r2))

        # Logarithmic
        model = fit_logarithmic(x, y)
        r2 = r_squared(y, model(x))
        results.append((i, "logarithmic", r2))

        # Polynomial
        model = fit_polynomial(x, y, degree)
        r2 = r_squared(y, model(x))
        results.append((i, f"polynomial_deg{degree}", r2))
        

    # Sort by R² descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results

def get_best_fit_per_feature(results):
    best_fits = {}

    for feature_idx, fit_type, r2 in results:
        # If feature not seen yet or current R² is better, update
        if feature_idx not in best_fits or r2 > best_fits[feature_idx][1]:
            best_fits[feature_idx] = (fit_type, r2)

    # Convert to sorted list by R² descending
    sorted_best = sorted(
        [(idx, fit_type, r2) for idx, (fit_type, r2) in best_fits.items()],
        key=lambda x: x[2],
        reverse=True
    )
    return sorted_best
# Map fit name -> fitter function (your functions return a predictor lambda)
FITTERS = {
    "linear":       fit_linear,
    "exponential":  fit_exponential,
    "reciprocal":   fit_reciprocal,
    "logarithmic":  fit_logarithmic,
    "polynomial_deg4":fit_polynomial
}


def transform_feature(x, fit_type):
    x = np.asarray(x)
    if fit_type == "linear":
        return x
    elif fit_type == "exponential":
        return x  # Exponential fit is applied to y, so x remains unchanged
    elif fit_type == "reciprocal":
        return 1.0 / np.clip(x, 1e-5, None)
    elif fit_type == "logarithmic":
        return np.log(np.clip(x, 1e-5, None))
    elif fit_type.startswith("polynomial_deg"):
        deg = int(fit_type.replace("polynomial_deg", ""))
        return np.column_stack([x**d for d in range(deg, 0, -1)])
    else:
        raise ValueError(f"Unknown fit type: {fit_type}")

def build_transformed_matrix(X, best_fits):
    n_samples, n_features = X.shape
    transformed_columns = []

    for j in range(n_features):
        xj = X[:, j]
        if not np.isfinite(xj).all() or np.allclose(xj, xj[0]):
            # If invalid or constant, use zeros
            transformed_columns.append(np.zeros((n_samples, 1)))
            continue

        fit_type = next((fit for idx, fit, _ in best_fits if idx == j), "linear")
        transformed = transform_feature(xj, fit_type)

        # Ensure 2D shape for stacking
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)

        transformed_columns.append(transformed)

    # Concatenate all transformed columns horizontally
    response_matrix = np.hstack(transformed_columns)
    return response_matrix
def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
def fit_and_predict(X_transformed, y_target):
    coeffs, _, _, _ = np.linalg.lstsq(X_transformed, y_target, rcond=None)
    y_pred = X_transformed @ coeffs
    return y_pred
def fit_and_predict_qr(X, y):
    # QR decomposition: X = Q @ R
    Q, R = np.linalg.qr(X)
    # Solve R @ beta = Q.T @ y
    beta = np.linalg.solve(R, Q.T @ y)
    return X @ beta
def fit_and_predict_svd(X, y):
    # SVD decomposition: X = U @ S @ V.T
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    # Compute pseudo-inverse: X⁺ = V.T.T @ diag(1/S) @ U.T
    S_inv = np.diag(1 / S)
    X_pinv = VT.T @ S_inv @ U.T
    beta = X_pinv @ y
    return X @ beta



# Violent
sorted_results_v = evaluate_fits(X_train, Y_violent)
best_fits_v = get_best_fit_per_feature(sorted_results_v)
response_matrix_v = build_transformed_matrix(X_train, best_fits_v)

y_pred_qr_v = fit_and_predict_qr(response_matrix_v, Y_violent)
y_pred_svd_v = fit_and_predict_svd(response_matrix_v, Y_violent)

mape_qr_v = mape(Y_violent, y_pred_qr_v)
mape_svd_v = mape(Y_violent, y_pred_svd_v)

# Non-Violent
sorted_results_nv = evaluate_fits(X_train, Y_non_violent)
best_fits_nv = get_best_fit_per_feature(sorted_results_nv)
response_matrix_nv = build_transformed_matrix(X_train, best_fits_nv)

y_pred_qr_nv = fit_and_predict_qr(response_matrix_nv, Y_non_violent)
y_pred_svd_nv = fit_and_predict_svd(response_matrix_nv, Y_non_violent)

mape_qr_nv = mape(Y_non_violent, y_pred_qr_nv)
mape_svd_nv = mape(Y_non_violent, y_pred_svd_nv)

print(f"MAPE (QR) for Violent: {mape_qr_v:.2f}%")
print(f"MAPE (SVD) for Violent: {mape_svd_v:.2f}%")
print(f"MAPE (QR) for Non-Violent: {mape_qr_nv:.2f}%")
print(f"MAPE (SVD) for Non-Violent: {mape_svd_nv:.2f}%")

Y_violent_log = np.log(np.clip(Y_violent, 1e-5, None))
Y_non_violent_log = np.log(np.clip(Y_non_violent, 1e-5, None))
y_pred_log_v = fit_and_predict(response_matrix_v, Y_violent_log)
y_pred_log_nv = fit_and_predict(response_matrix_nv, Y_non_violent_log)
y_pred_v = np.exp(y_pred_log_v)
y_pred_nv = np.exp(y_pred_log_nv)
mape_log_v = mape(Y_violent, y_pred_v)
mape_log_nv = mape(Y_non_violent, y_pred_nv)

print(f"MAPE (log-transformed) for Violent: {mape_log_v:.2f}%")
print(f"MAPE (log-transformed) for Non-Violent: {mape_log_nv:.2f}%")
