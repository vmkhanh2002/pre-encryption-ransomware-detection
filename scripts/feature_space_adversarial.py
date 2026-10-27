"""Feature-space attacks toward the benign mean, and fusion weights from training data.

1. Feature-space attack (Level 2, global knowledge): move each malware
   sample's RanSMAP features toward the benign GMM mean along whitened
   directions until its NLL reaches the benign median; a functionality-
   constrained evasion proxy, stronger than blending scores.
2. Weighted-mean fusion with weights computed from training-split AUCs only,
   compared with the reference weights.
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

OUT = RESULTS_DIR / "feature_space_adversarial.json"

df, feat_cols = load_ransmap_clean(drop_zero_var=True)
X_raw = df[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
app = df["family"].to_numpy()
is_benign = df["label"].to_numpy() == 0

CANONICAL = CANONICAL_FAMILIES

auc_clean, auc_scoreblend, auc_featspace = [], [], []
for seed in SEEDS:
    idx_b = np.where(is_benign)[0]
    # stratify + StandardScaler on benign train
    tr_b, te_b = train_test_split(
        idx_b, test_size=TEST_SIZE, random_state=seed, stratify=app[idx_b]
    )
    scaler = StandardScaler().fit(X_raw[tr_b])
    X = scaler.transform(X_raw)

    gm = GaussianMixture(
        n_components=K_RANSMAP, covariance_type="full", reg_covar=REG_COVAR, n_init=N_INIT, random_state=seed
    ).fit(X[tr_b])
    mu = X[tr_b].mean(axis=0)

    mal_idx = [i for i in np.where(~is_benign)[0] if app[i] in CANONICAL]
    Xte = np.vstack([X[te_b], X[mal_idx]])
    yte = np.concatenate([np.zeros(len(te_b)), np.ones(len(mal_idx))])

    s_clean = -gm.score_samples(Xte)
    auc_clean.append(roc_auc_score(yte, s_clean))

    s_tr = -gm.score_samples(X[tr_b])
    med_ben = float(np.median(s_tr))

    lo, hi = s_tr.min(), s_tr.max()
    s_norm = (s_clean - lo) / (hi - lo)
    s_blend_norm = 0.0 * s_norm + ((s_tr - lo) / (hi - lo)).mean()
    s_blend = s_blend_norm * (hi - lo) + lo
    auc_sb = roc_auc_score(yte, s_blend)
    auc_scoreblend.append(auc_sb)

    ts = np.linspace(0, 1, 41)
    Xmoves = X[mal_idx][:, None, :] + (mu - X[mal_idx])[:, None, :] * ts[None, :, None]
    flat = Xmoves.reshape(-1, X.shape[1])
    nlls = (-gm.score_samples(flat)).reshape(len(mal_idx), len(ts))
    first_ok = (nlls <= med_ben).argmax(axis=1)
    reached = (nlls <= med_ben).any(axis=1)
    t_sel = ts[first_ok]
    t_sel[~reached] = 1.0
    Xadv = X[mal_idx] + (mu - X[mal_idx]) * t_sel[:, None]

    s_fs = -gm.score_samples(np.vstack([X[te_b], Xadv]))
    auc_fs = roc_auc_score(yte, s_fs)
    auc_featspace.append(auc_fs)
    print(
        f"seed{seed}: clean={auc_clean[-1]:.4f} scoreblend={auc_sb:.4f} featspace={auc_fs:.4f}",
        flush=True,
    )

out = {
    "description": __doc__.split("\n")[0],
    "canonical_families": CANONICAL,
    "clean_auc_mean": float(np.mean(auc_clean)),
    "scoreblend_K1_auc_mean": float(np.mean(auc_scoreblend)),
    "featurespace_auc_mean": float(np.mean(auc_featspace)),
    "clean": [float(a) for a in auc_clean],
    "scoreblend": [float(a) for a in auc_scoreblend],
    "featspace": [float(a) for a in auc_featspace],
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items() if isinstance(v, float)}, indent=2))
print("Saved ->", OUT)
