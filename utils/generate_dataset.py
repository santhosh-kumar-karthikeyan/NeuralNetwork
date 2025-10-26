import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os

# === Configuration ===
n_samples = 5000       # Number of data points
n_features = 10        # Number of input features
n_classes = 2          # Binary classification
random_state = 42      # For reproducibility

# Get absolute path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "assets")

# === Generate dataset ===
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_features,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=2.0,          # Separation for linear separability
    n_classes=n_classes,
    random_state=random_state
)

# === Create output directory ===
os.makedirs(output_dir, exist_ok=True)

# === Save as CSV ===
df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
df["label"] = y
csv_path = os.path.join(output_dir, "linearly_separable_data.csv")
df.to_csv(csv_path, index=False)

# === Save as NumPy binary ===
np.save(os.path.join(output_dir, "X.npy"), X)
np.save(os.path.join(output_dir, "y.npy"), y)

# === Verification ===
print(f"Dataset created with shape: X={X.shape}, y={y.shape}")
print(f"Saved to folder: {output_dir}")
print("\nSample data:\n", df.head())
