"""Bounded feature-space attacks toward the benign mean (Level 2).

The benign mean itself has an NLL far above the benign median: the GMM's
high-density regions are narrow and multi-modal, and the mean lies between
modes. This script reports the NLL reduction an attacker achieves for each
displacement budget toward the mean, showing that bounded moves in that
direction never reach the benign range (AUC stays near the clean value).
"""
import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import CANONICAL_FAMILIES, K_RANSMAP, N_INIT, REG_COVAR, RESULTS_DIR, SEEDS, TEST_SIZE  # noqa: E402
from utils import load_ransmap_clean  # noqa: E402

OUT = RESULTS_DIR / "whitebox_adversarial.json"

CANON_R = CANONICAL_FAMILIES
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

dfr, fr = load_ransmap_clean(drop_zero_var=True)
Xr_raw = dfr[fr].fillna(0.0).to_numpy(np.float64)
is_ben = dfr["label"].to_numpy() == 0
famr = dfr["family"].to_numpy()

results = []
for seed in SEEDS:
    idx_ben = np.where(is_ben)[0]
    # stratify + StandardScaler on benign train
    tr_b, te_b = train_test_split(
        idx_ben, test_size=TEST_SIZE, random_state=seed, stratify=famr[idx_ben]
    )
    scaler = StandardScaler().fit(Xr_raw[tr_b])
    Xr = scaler.transform(Xr_raw)

    gm = GaussianMixture(
        n_components=K_RANSMAP, covariance_type="full", reg_covar=REG_COVAR, n_init=N_INIT, random_state=seed
    ).fit(Xr[tr_b])
    mal_idx = [i for i in np.where(~is_ben)[0] if famr[i] in CANON_R]
    Xorig = Xr[mal_idx]
    mu = Xr[tr_b].mean(axis=0)  # zero vector in standardized space
    s_tr = -gm.score_samples(Xr[tr_b])
    med_ben = float(np.median(s_tr))
    thr5 = np.sort(s_tr)[-int(np.ceil(0.05 * len(tr_b)))]
    s_ben_te = -gm.score_samples(Xr[te_b])

    row = {"seed": seed, "benign_median_nll": round(med_ben, 1)}
    for alpha in ALPHAS:
        Xadv = Xorig + (mu - Xorig) * alpha
        s_mal = -gm.score_samples(Xadv)
        auc = float(
            roc_auc_score(
                np.concatenate([np.zeros(len(te_b)), np.ones(len(Xadv))]),
                np.concatenate([s_ben_te, s_mal]),
            )
        )
        row[f"a{alpha}"] = {
            "auc": round(auc, 4),
            "tpr5": round(float((s_mal >= thr5).mean()), 4),
            "nll_mean": round(float(s_mal.mean()), 1),
            "frac_reaching_benign_median": round(float((s_mal <= med_ben).mean()), 4),
        }
    results.append(row)

agg = {}
for a in ALPHAS:
    k = f"a{a}"
    agg[k] = {
        "auc_mean": round(float(np.mean([r[k]["auc"] for r in results])), 4),
        "tpr5_mean": round(float(np.mean([r[k]["tpr5"] for r in results])), 4),
        "nll_mean": round(float(np.mean([r[k]["nll_mean"] for r in results])), 1),
        "frac_evasion_success": round(
            float(np.mean([r[k]["frac_reaching_benign_median"] for r in results])), 4
        ),
    }

out = {
    "description": __doc__.split("\n")[0],
    "benign_median_nll_mean": round(float(np.mean([r["benign_median_nll"] for r in results])), 1),
    "aggregate": agg,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(agg, indent=2))
print("Saved ->", OUT)
