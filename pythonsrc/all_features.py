import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import path, makedirs
from train import X_train, Y_violent, Y_non_violent

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


if __name__ == "__main__":
    col_names = load_column_names("./proj1files/columns.txt")

    # Create a directory to save plots if needed
    output_dir = "./media/feature_plots"
    if not path.exists(output_dir):
        makedirs(output_dir)

    for idx in range(0, 78):  # Assuming 78 features
        feature_name = col_names[idx]
        feature_data = X_train[:, idx]

        # Create subplots: 1 row, 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Feature: {feature_name}", fontsize=14)

        # Plot X_train vs Y_violent
        sns.scatterplot(x=feature_data, y=Y_violent, ax=axes[0], color='red',s=3)
        axes[0].set_title("Violent")
        axes[0].set_xlabel(feature_name)
        axes[0].set_ylabel("Y_violent")

        # Plot X_train vs Y_non_violent
        sns.scatterplot(x=feature_data, y=Y_non_violent, ax=axes[1], color='blue',s=3)
        axes[1].set_title("Non-Violent")
        axes[1].set_xlabel(feature_name)
        axes[1].set_ylabel("Y_non_violent")

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(path.join(output_dir, f"{idx}_subplot.png"))
        plt.close()

