"""Exact TPR at low FPR by threshold counting instead of ROC interpolation.

At the FPR values a finite test set can actually resolve (multiples of
1/n_benign), count true and false positives at the k-th highest benign
score rather than interpolating the ROC curve. Also reports the smallest
FPR each test set can support (paper, Section 4.2, 0.1%-FPR paragraph).
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
from config import K_MLRAN, K_RANSMAP, N_INIT, REG_COVAR, RESULTS_DIR, SEEDS, TEST_SIZE  # noqa: E402
from utils import load_mlran, load_ransmap_clean  # noqa: E402

OUT = RESULTS_DIR / "exact_fpr_counts.json"

results = {
    "description": __doc__.split("\n")[0],
    "channels": {},
}

# ---------- MLRan (split benign groups only, evaluate on 100% malware) ----------
dfm, fm = load_mlran()
Xm = dfm[fm].fillna(0.0).to_numpy(np.float32)
ym = dfm["label"].to_numpy()
Xm_ben = Xm[ym == 0]
Xm_mal = Xm[ym == 1]
groups_ben = np.array([hashlib.md5(r.tobytes()).hexdigest() for r in Xm_ben])
uniq_g_ben = np.unique(groups_ben)

mlran_res = []
for seed in SEEDS:
    tr_g, _ = train_test_split(uniq_g_ben, test_size=TEST_SIZE, random_state=seed)
    mtr = np.isin(groups_ben, tr_g)
    gm = GaussianMixture(
        n_components=K_MLRAN, covariance_type="diag", reg_covar=REG_COVAR, n_init=N_INIT, random_state=seed
    ).fit(Xm_ben[mtr])
    s_ben = -gm.score_samples(Xm_ben[~mtr])
    s_mal = -gm.score_samples(Xm_mal)
    nb = len(s_ben)
    row = {"n_ben_test": int(nb), "n_mal_test": int(len(s_mal))}
    for q in (0.001, 0.01, 0.05):
        k = max(1, int(np.ceil(q * nb)))
        thr = np.sort(s_ben)[-k]
        fpr_exact = float((s_ben >= thr).mean())
        tpr_exact = float((s_mal >= thr).mean())
        row[f"q{q}"] = {
            "k_threshold": int(k),
            "exact_fpr": round(fpr_exact, 5),
            "exact_tpr": round(tpr_exact, 4),
        }
    mlran_res.append(row)

results["channels"]["mlran"] = {
    "paper_claim": "TPR=0.2957 @ 0.1%FPR",
    "per_seed": mlran_res,
    "min_supportable_fpr": round(
        float(np.mean([1 / r["n_ben_test"] for r in mlran_res])), 5
    ),
}

# ---------- RanSMAP (stratify + StandardScaler + 29 clean features) ----------
dfr, fr = load_ransmap_clean(drop_zero_var=True)
Xr_raw = dfr[fr].fillna(0.0).to_numpy(np.float64)
is_ben = dfr["label"].to_numpy() == 0
fam_r = dfr["family"].to_numpy()

ransmap_res = []
for seed in SEEDS:
    idx_ben = np.where(is_ben)[0]
    tr_b, te_b = train_test_split(
        idx_ben, test_size=TEST_SIZE, random_state=seed, stratify=fam_r[idx_ben]
    )
    mal_idx = np.where(~is_ben)[0]
    scaler = StandardScaler().fit(Xr_raw[tr_b])
    Xr_tr = scaler.transform(Xr_raw[tr_b])
    Xr_te_ben = scaler.transform(Xr_raw[te_b])
    Xr_mal = scaler.transform(Xr_raw[mal_idx])

    gm = GaussianMixture(
        n_components=K_RANSMAP, covariance_type="full", reg_covar=REG_COVAR, n_init=N_INIT, random_state=seed
    ).fit(Xr_tr)
    s_ben = -gm.score_samples(Xr_te_ben)
    s_mal = -gm.score_samples(Xr_mal)
    nb = len(s_ben)
    row = {"n_ben_test": int(nb), "n_mal_test": int(len(s_mal))}
    for q in (0.001, 0.01, 0.05):
        k = max(1, int(np.ceil(q * nb)))
        thr = np.sort(s_ben)[-k]
        row[f"q{q}"] = {
            "k_threshold": int(k),
            "exact_fpr": round(float((s_ben >= thr).mean()), 5),
            "exact_tpr": round(float((s_mal >= thr).mean()), 4),
        }
    ransmap_res.append(row)

results["channels"]["ransmap"] = {
    "paper_claim": "TPR=0.800 @ 0.1%FPR",
    "per_seed": ransmap_res,
    "min_supportable_fpr": round(
        float(np.mean([1 / r["n_ben_test"] for r in ransmap_res])), 5
    ),
}

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)

for ch in ("mlran", "ransmap"):
    per = results["channels"][ch]["per_seed"]
    for q in ("q0.001", "q0.01", "q0.05"):
        tprs = [p[q]["exact_tpr"] for p in per]
        fprs = [p[q]["exact_fpr"] for p in per]
        print(f"{ch} {q}: mean TPR={np.mean(tprs):.4f} mean FPR={np.mean(fprs):.4f}")
print("Saved ->", OUT)
