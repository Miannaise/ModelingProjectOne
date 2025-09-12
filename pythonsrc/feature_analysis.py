import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from train import X_train, Y_violent

# --- Utility Functions ---

def covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute covariance between two arrays."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"Shapes do not match: x.shape={x.shape}, y.shape={y.shape}")
    return np.mean((x - x.mean()) * (y - y.mean()))

def load_column_names(filepath: str) -> list[str]:
    """Load column names from a text file."""
    with open(filepath, 'r') as f:
        return f.read().splitlines()

def compute_top_covariances(X: np.ndarray, y: np.ndarray, start_col: int, end_col: int, top_n: int = 5) -> list[tuple[float, int]]:
    """Compute and return top N features with highest absolute covariance."""
    cov_list = [(covariance(X[:, col], y), col) for col in range(start_col, end_col)]
    cov_list.sort(key=lambda x: abs(x[0]), reverse=True)
    return cov_list[:top_n]

def plot_and_save_feature(X: np.ndarray, y: np.ndarray, feature_idx: int, feature_name: str, cov_value: float, output_dir: str = "./media"):
    """Generate and save a scatter plot for a given feature."""
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=X[:, feature_idx], y=y)
    plt.xlabel(f"Feature: {feature_name}")
    plt.ylabel("Y_violent")
    plt.title(f"Scatter Plot\nCovariance: {cov_value:.2f}")
    plt.tight_layout()

    filename = f"{output_dir}/plot{feature_idx}.png"
    if not path.exists(filename):
        plt.savefig(filename)
    else:
        print(f"{filename} already exists.")

# --- Main Execution ---

if __name__ == "__main__":
    col_names = load_column_names("./proj1files/columns.txt")
    top_features = compute_top_covariances(X_train, Y_violent, start_col=1, end_col=78, top_n=5)

    for cov, idx in top_features:
        feature_name = col_names[idx]
        print(f"Feature: {feature_name} | Covariance: {cov:.2f}")
        plot_and_save_feature(X_train, Y_violent, idx, feature_name, cov)
