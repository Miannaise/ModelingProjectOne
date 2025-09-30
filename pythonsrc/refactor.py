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

# def fit_exponential(x, y):
#     y = np.clip(y, 1e-5, None)
#     log_y = np.log(y)
#     A = np.vstack([x, np.ones_like(x)]).T
#     coeffs = np.linalg.lstsq(A, log_y, rcond=None)[0]
#     return lambda x_: np.exp(coeffs[0] * x_ + coeffs[1])

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

        # # Exponential
        # model = fit_exponential(x, y)
        # r2 = r_squared(y, model(x))
        # results.append((i, "exponential", r2))

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
    # "exponential":  fit_exponential,
    "reciprocal":   fit_reciprocal,
    "logarithmic":  fit_logarithmic,
    "polynomial_deg4":fit_polynomial
}


def transform_feature(x, fit_type):
    x = np.asarray(x)
    if fit_type == "linear":
        return x
    # elif fit_type == "exponential":
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


# Violent
sorted_results_v = evaluate_fits(X_train, Y_violent)
best_fits_v = get_best_fit_per_feature(sorted_results_v)
print(np.shape(best_fits_v))
response_matrix_v = build_transformed_matrix(X_train, best_fits_v)
beta_hat_v,*_=np.linalg.lstsq(response_matrix_v, Y_violent, rcond=None)
Y_violent_pred=  response_matrix_v @ beta_hat_v
mape_v = mape(Y_violent, Y_violent_pred)
Y_violent_log = np.log(np.clip(Y_violent, 1e-5, None))
beta_hat_v_log,*_=np.linalg.lstsq(response_matrix_v, Y_violent_log, rcond=None)
Y_violent_exp_pred=  np.exp(response_matrix_v @ beta_hat_v_log)
mape_v_exp = mape(Y_violent, Y_violent_exp_pred)
print(f"Violent Crime MAPE (direct): {mape_v:.2f}")
print(f"Violent Crime MAPE (log fit on y): {mape_v_exp:.2f}")

#load test data
testing_data=scio.loadmat("./proj1files/testingdata.mat")
testing_data=testing_data["testingdata"]
X_test=testing_data[:,0:78]
Y_violent_test=testing_data[:,79]
Y_non_violent_test=testing_data[:,80]
# X_test = np.insert(X_test, 0, np.ones((X_test.shape[0],)), axis=1)
response_matrix_v_test = build_transformed_matrix(X_test, best_fits_v)
Y_violent_test_pred = response_matrix_v_test @ beta_hat_v
mape_v_test = mape(Y_violent_test, Y_violent_test_pred)
Y_violent_test_exp_pred = np.exp(response_matrix_v_test @ beta_hat_v_log)
mape_v_test_exp = mape(Y_violent_test, Y_violent_test_exp_pred)
print(f"Violent Crime Test MAPE (direct): {mape_v_test:.22f}")
print(f"Violent Crime Test MAPE (log fit on y): {mape_v_test_exp:.2f}")
# Non Violent
sorted_results_nv = evaluate_fits(X_train, Y_non_violent)
best_fits_nv = get_best_fit_per_feature(sorted_results_nv)
response_matrix_nv = build_transformed_matrix(X_train, best_fits_nv)
beta_hat_nv,*_=np.linalg.lstsq(response_matrix_nv, Y_non_violent, rcond=None)
Y_non_violent_pred=  response_matrix_nv @ beta_hat_nv
mape_nv = mape(Y_non_violent, Y_non_violent_pred)
Y_non_violent_log = np.log(np.clip(Y_non_violent, 1e-5, None))
beta_hat_nv_log,*_=np.linalg.lstsq(response_matrix_nv, Y_non_violent_log, rcond=None)
Y_non_violent_exp_pred=  np.exp(response_matrix_nv @ beta_hat_nv_log)
mape_nv_exp = mape(Y_non_violent, Y_non_violent_exp_pred)
print(f"Non Violent Crime MAPE (direct): {mape_nv:.2f}")
print(f"Non Violent Crime MAPE (log fit on y): {mape_nv_exp:.2f}")
response_matrix_nv_test = build_transformed_matrix(X_test, best_fits_nv)
Y_non_violent_test_pred = response_matrix_nv_test @ beta_hat_nv
mape_nv_test = mape(Y_non_violent_test, Y_non_violent_test_pred)
Y_non_violent_test_exp_pred = np.exp(response_matrix_nv_test @ beta_hat_nv_log)
mape_nv_test_exp = mape(Y_non_violent_test, Y_non_violent_test_exp_pred)
print(f"Non Violent Crime Test MAPE (direct): {mape_nv_test:.2f}")
print(f"Non Violent Crime Test MAPE (log fit on y): {mape_nv_test_exp:.2f}")
