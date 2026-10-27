"""Single source of paths and experiment settings.

Every script takes its file locations and shared settings from here, so no
script hard-codes a path, a hyperparameter shared with other scripts, or a
number computed by another script.

Paths
-----
The repository root is the nearest ancestor of this file that contains
`requirements.txt` and `src/config.py`. Environment variables override
every default:

    PERD_REPO_ROOT     repository root
    PERD_MLRAN_CSV     processed MLRan feature table
    PERD_RANSMAP_CSV   processed RanSMAP feature table
    PERD_RESULTS_DIR   where result JSON files are written and read
    PERD_FIGURES_DIR   where figures are written
    RANSMAP_KAGGLE_ZIP raw RanSMAP Kaggle archive (feature extraction only)
"""
import json
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT_MARKERS = ("requirements.txt", "src/config.py")


def _find_repo_root(start):
    for d in (start, *start.parents):
        if all((d / m).is_file() for m in _ROOT_MARKERS):
            return d
    raise RuntimeError(f"No ancestor of {start} contains {', '.join(_ROOT_MARKERS)}; "
                       "set PERD_REPO_ROOT to the repository root.")


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


REPO_ROOT = _env_path("PERD_REPO_ROOT", _find_repo_root(_HERE))
DATA_DIR = REPO_ROOT / "data" / "processed"
MLRAN_CSV = _env_path("PERD_MLRAN_CSV", DATA_DIR / "mlran_analysis_features.csv")
RANSMAP_CSV = _env_path("PERD_RANSMAP_CSV", DATA_DIR / "ransmap_analysis_features.csv")
RESULTS_DIR = _env_path("PERD_RESULTS_DIR", REPO_ROOT / "experiments" / "ano_baseline")
FIGURES_DIR = _env_path("PERD_FIGURES_DIR", REPO_ROOT / "figures")


def ransmap_kaggle_zip():
    """Raw RanSMAP Kaggle archive; required only by full feature extraction."""
    value = os.environ.get("RANSMAP_KAGGLE_ZIP")
    if not value:
        raise SystemExit("Set RANSMAP_KAGGLE_ZIP to the RanSMAP dataset archive downloaded "
                         "from Kaggle (hiranomanabu/ransmap-2024-ransomware-behavioral-features).")
    return Path(value).expanduser().resolve()


# -- Experiment settings shared by several scripts ----------------------------

SEEDS = list(range(10))
CANONICAL_FAMILIES = ["Conti", "Ryuk", "LockBit", "REvil", "WannaCry"]
TEST_SIZE = 0.2

# Per-channel benign-only GMMs (Section 3.2-3.3 of the paper).
K_MLRAN = 3            # diagonal covariance
COV_MLRAN = "diag"
K_RANSMAP = 5          # full covariance; BIC minimum, = number of benign process types
COV_RANSMAP = "full"
REG_COVAR = 1e-3       # per-channel GMMs with K > 1
REG_COVAR_SINGLE = 1e-6  # single-component GMMs
N_INIT = 5

# Meta-GMM over the 2-D score cloud (learned fusion).
META_K_CANDIDATES = [1, 2, 3]
META_REG_COVAR = 1e-6


# -- Values produced by other scripts ------------------------------------------

def load_result(name):
    """Load a result JSON written by another script; fail clearly if missing."""
    path = RESULTS_DIR / name
    if not path.is_file():
        raise SystemExit(f"{path} not found: run the script that produces {name} first "
                         "(order in README.md).")
    with open(path) as f:
        return json.load(f)


def selected_meta_k(variant="oof"):
    """Meta-GMM component count chosen by BIC in learned_fusion.py.

    variant="oof": calibration on out-of-fold benign training scores (used by
    the adversarial experiments); variant="split75": calibration on half of
    the benign test pool (paper, Table 3).
    """
    sel = load_result("learned_fusion.json")["meta_k_selection"]
    key = {"oof": "selected_k", "split75": "selected_k_split75_noreplace"}[variant]
    return int(sel[key])


def fused_result_key(variant="split75"):
    """Key of the fused system in learned_fusion.json for a calibration variant."""
    k = selected_meta_k(variant)
    return f"meta_K{k}" if variant == "oof" else f"meta_K{k}_split75_noreplace"
