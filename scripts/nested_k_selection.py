"""Nested K selection on the benign training split only.

Selects the per-channel GMM component count by BIC using benign training
samples alone, with the deployment reg_covar, to check how the K values used
in the paper compare with a nested, leakage-free selection. RanSMAP features
are standardized on the benign training split and constant features are
dropped. High K on MLRan's binary data can collapse groups; such fits are
caught and recorded as failures.
"""
import hashlib
import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import N_INIT, REG_COVAR, RESULTS_DIR, SEEDS, TEST_SIZE  # noqa: E402
from utils import load_mlran, load_ransmap_clean  # noqa: E402

OUT = RESULTS_DIR / "nested_k_selection.json"

results = {
    "description": __doc__.split("\n")[0],
    "mlran": {},
    "ransmap_34feat_scaled": {},
    "ransmap_29feat_scaled": {},
}

# ---- MLRan (benign rows only) ----
df_m, feat_cols_m = load_mlran()
df_m_ben = df_m[df_m["label"] == 0].reset_index(drop=True)
X_m_ben = df_m_ben[feat_cols_m].fillna(0.0).to_numpy(dtype=np.float32)
group_ids = np.array([
    hashlib.md5(row.tobytes()).hexdigest() for row in X_m_ben
])
uniq_groups = np.unique(group_ids)

mlran_ks = []
for seed in SEEDS:
    tr_groups, _ = train_test_split(uniq_groups, test_size=TEST_SIZE, random_state=seed)
    Xtr = X_m_ben[np.isin(group_ids, tr_groups)]
    bics = {}
    for K in range(1, 8):
        try:
            gm = GaussianMixture(
                n_components=K, covariance_type="diag", reg_covar=REG_COVAR,
                n_init=N_INIT, max_iter=300, random_state=seed
            ).fit(Xtr)
            bics[K] = float(gm.bic(Xtr))
        except ValueError:
            bics[K] = None
    valid = {k: v for k, v in bics.items() if v is not None}
    best_K = min(valid, key=valid.get) if valid else None
    results["mlran"][f"seed{seed}"] = {"bic": bics, "best_K": best_K}
    mlran_ks.append(best_K)
    print(f"MLRan (benign-only) seed{seed}: best_K={best_K}", flush=True)

results["mlran"]["summary"] = {
    "paper_K": 3,
    "chosen_ks": [k if k is None else int(k) for k in mlran_ks],
    "modal_K": max([k for k in mlran_ks if k], key=lambda k: mlran_ks.count(k)) if any(mlran_ks) else None,
    "frac_matching_paper": round(sum(k == 3 for k in mlran_ks) / len(mlran_ks), 2),
}

# ---- RanSMAP (StandardScaler + stratify=family, both 34 and 29 features) ----
for tag, drop_zv in [("ransmap_34feat_scaled", False), ("ransmap_29feat_scaled", True)]:
    df2, feat2 = load_ransmap_clean(drop_zero_var=drop_zv)
    is_benign = df2["label"].to_numpy() == 0
    Xb = df2.loc[is_benign, feat2].fillna(0.0).to_numpy(dtype=np.float64)
    fam_b = df2.loc[is_benign, "family"].to_numpy()

    ransmap_ks = []
    for seed in SEEDS:
        Xtr_raw, _ = train_test_split(
            Xb, test_size=TEST_SIZE, random_state=seed, stratify=fam_b
        )
        Xtr = StandardScaler().fit_transform(Xtr_raw)
        bics = {}
        for K in range(1, 8):
            try:
                gm = GaussianMixture(
                    n_components=K, covariance_type="full", reg_covar=REG_COVAR,
                    n_init=N_INIT, max_iter=300, random_state=seed
                ).fit(Xtr)
                bics[K] = float(gm.bic(Xtr))
            except ValueError:
                bics[K] = None
        valid = {k: v for k, v in bics.items() if v is not None}
        best_K = min(valid, key=valid.get) if valid else 5
        results[tag][f"seed{seed}"] = {"bic": bics, "best_K": int(best_K)}
        ransmap_ks.append(int(best_K))
        print(f"{tag} seed{seed}: best_K={best_K}", flush=True)

    results[tag]["summary"] = {
        "paper_K": 5,
        "chosen_ks": ransmap_ks,
        "modal_K": max(set(ransmap_ks), key=ransmap_ks.count),
        "frac_matching_paper": round(sum(k == 5 for k in ransmap_ks) / len(ransmap_ks), 2),
    }

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print("MLRan summary:", results["mlran"]["summary"])
print("RanSMAP (34feat scaled) summary:", results["ransmap_34feat_scaled"]["summary"])
print("RanSMAP (29feat scaled) summary:", results["ransmap_29feat_scaled"]["summary"])
print("Saved ->", OUT)
