"""Shared utilities for PERD Fusion System.

Provides consistent model fitting, data loading, and normalization
functions used across all experiment scripts.
"""

from config import MLRAN_CSV, RANSMAP_CSV, RESULTS_DIR, REG_COVAR, REG_COVAR_SINGLE, N_INIT  # noqa: E402
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture

__version__ = "1.0.0"
__all__ = [
    "MLRAN_PATH", "RANSMAP_PATH", "OUTPUT_DIR",
    "MLRAN_EXCLUDE", "RANSMAP_EXCLUDE",
    "fit_gm", "minmax_norm", "pct_rank",
    "load_mlran", "load_ransmap", "validate_columns",
]

# -- Paths --------------------------------------------------------------------

# Repository root is one level up from src/. Processed CSVs live under data/processed/.
MLRAN_PATH = MLRAN_CSV
RANSMAP_PATH = RANSMAP_CSV
OUTPUT_DIR = RESULTS_DIR

# -- Column exclusion sets (single source of truth) ---------------------------

MLRAN_EXCLUDE = {"label", "family", "sample_id", "sha256", "split", "STRING:labels", "dataset"}
RANSMAP_EXCLUDE = {"label", "family", "trace_path", "dataset"}

# -- Model utilities ----------------------------------------------------------


def fit_gm(X, n_components, covariance_type, seed=42, reg_covar=None, max_iter=300):
    """Fit a GaussianMixture with consistent hyperparameters.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    n_components : int
    covariance_type : one of {"full", "diag", "spherical", "tied"}
    seed : int
        Passed to ``random_state``.
    reg_covar : float or None
        Defaults to 1e-3 for K>1, 1e-6 for K=1.
    max_iter : int
        EM maximum iterations.

    Returns
    -------
    fitted : GaussianMixture
    """
    if reg_covar is None:
        reg_covar = REG_COVAR if n_components > 1 else REG_COVAR_SINGLE
    n_init = N_INIT if n_components > 1 else 1
    return GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=seed,
        max_iter=max_iter,
    ).fit(X)


# -- Normalization ------------------------------------------------------------


def minmax_norm(ref, x):
    """MinMax-normalize *x* using the range of *ref*.

    Returns **zeros** (not the raw input) when *ref* is nearly constant
    (``hi - lo < 1e-10``).  The old behaviour of returning the un-normalized
    *x* silently produced out-of-range values that corrupted downstream
    product fusion.
    """
    lo, hi = float(ref.min()), float(ref.max())
    rng = hi - lo
    if rng < 1e-10:
        return np.nan_to_num(np.full_like(x, 0.0, dtype=float), nan=0.0)
    return (x - lo) / rng


def pct_rank(ref_scores, query_scores):
    """Percentile rank of each query score within the reference distribution.

    Returns values in [0, 100] where 100 means *query* is more anomalous
    than every reference sample.

    Uses ``np.searchsorted`` for O(n log n) performance instead of the
    previous O(n^2) list comprehension. ``side="left"`` counts strictly-less
    reference scores, matching the original ``mean(ref < q)`` exactly (so the
    numeric result is unchanged, including under ties).
    """
    sorted_ref = np.sort(ref_scores)
    return np.searchsorted(sorted_ref, query_scores, side="left") / len(sorted_ref) * 100.0


# -- Data loading -------------------------------------------------------------


def validate_columns(feat_cols, expected_dim, name):
    """Assert feature count matches expectation. Call after column filtering."""
    actual = len(feat_cols)
    if actual != expected_dim:
        raise ValueError(
            f"[{name}] expected {expected_dim} feature columns, got {actual}. "
            f"Check CSV schema or exclusion set."
        )


def load_mlran():
    """Load the MLRan dataset and compute feature columns.

    Returns
    -------
    df : DataFrame with all rows and columns
    feat_cols : list of feature-column names (excludes metadata columns)
    """
    df = pd.read_csv(MLRAN_PATH, index_col=0)
    feat_cols = [c for c in df.columns if c not in MLRAN_EXCLUDE]
    return df, feat_cols


def load_ransmap():
    """Load the RanSMAP dataset and compute feature columns.

    Returns
    -------
    df : DataFrame with all rows and columns
    feat_cols : list of numeric feature-column names (excludes metadata columns)
    """
    df = pd.read_csv(RANSMAP_PATH)
    num_cols = set(df.select_dtypes(include="number").columns)
    feat_cols = sorted(c for c in df.columns if c in num_cols and c not in RANSMAP_EXCLUDE)
    return df, feat_cols
