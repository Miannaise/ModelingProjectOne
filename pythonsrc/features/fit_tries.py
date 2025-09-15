import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import path, makedirs

from train import X_train, Y_violent

# --- Utility Functions ---

def ensure_dir(directory: str):
    if not path.exists(directory):
        makedirs(directory)

def save_plot(fig, idx: int):
    folder = "./media/regplots"
    ensure_dir(folder)
    filename = f"{folder}/plot{idx}.png"
    if not path.exists(filename):
        fig.savefig(filename)
    else:
        print(f"{filename} already exists.")
    plt.close(fig)

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


if __name__ == "__main__":
    for idx in range(1, 78):
        x = X_train[:, idx]
        y = Y_violent

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=x, y=y, ax=ax, alpha=0.5)

        x_sorted = np.sort(x)

        # Fit and plot each model
        models = {
            "Linear": fit_linear(x, y),
            "Exponential": fit_exponential(x, y),
            "1/x": fit_reciprocal(x, y),
            "Logarithmic": fit_logarithmic(x, y),
            "Quadratic": fit_polynomial(x, y, 2),
            "Cubic": fit_polynomial(x, y, 3),
        }

        for label, model in models.items():
            y_pred = model(x)
            r2 = r_squared(y, y_pred)
            ax.plot(x_sorted, model(x_sorted), label=f"{label} (R²={r2:.3f})")

        ax.set_title(f"Feature {idx} vs Y_violent")
        ax.set_xlabel(f"Feature {idx}")
        ax.set_ylabel("Y_violent")
        ax.legend()
        plt.tight_layout()
        save_plot(fig, idx)
