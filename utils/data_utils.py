"""
Utilities for data handling.
"""
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    return pd.read_csv(path)