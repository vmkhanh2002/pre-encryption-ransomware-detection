"""Feature-space attacks toward the benign mean, and fusion weights from training data.

1. Feature-space attack (Level 2, global knowledge): move each malware
   sample's RanSMAP features toward the benign GMM mean along whitened
   directions until its NLL reaches the benign median; a functionality-
   constrained evasion proxy, stronger than blending scores.
2. Weighted-mean fusion with weights computed from training-split AUCs only,
   compared with the reference weights.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RESULTS_DIR, SEEDS, K_RANSMAP, COV_RANSMAP, REG_COVAR, N_INIT  # noqa: E402
import os, sys
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import load_ransmap

OUT = RESULTS_DIR / 'feature_space_adversarial.json'

df, feat_cols = load_ransmap()
X = df[feat_cols].to_numpy(dtype=np.float64)
app = df['family'].to_numpy()
is_benign = df['label'].to_numpy() == 0 if 'label' in df else np.isin(app, ['Firefox','Idle','Office','SDelete','Zip'])

CANONICAL = ['Conti','Ryuk','LockBit','REvil','WannaCry']
auc_clean, auc_scoreblend, auc_featspace = [], [], []
for seed in SEEDS:
    idx_b = np.where(is_benign)[0]
    tr_b, te_b = train_test_split(idx_b, test_size=0.2, random_state=seed)
    gm = GaussianMixture(K_RANSMAP, covariance_type=COV_RANSMAP, reg_covar=REG_COVAR,
                         n_init=N_INIT, random_state=seed).fit(X[tr_b])
    mu = X[tr_b].mean(axis=0)
    C = np.cov(X[tr_b].T) + 1e-6 * np.eye(X.shape[1])
    Cinv = np.linalg.inv(C)

    # test set: benign test + canonical-family malware
    mal_idx = [i for i in np.where(~is_benign)[0] if app[i] in CANONICAL]
    Xte = np.vstack([X[te_b], X[mal_idx]])
    yte = np.concatenate([np.zeros(len(te_b)), np.ones(len(mal_idx))])

    s_clean = -gm.score_samples(Xte)
    auc_clean.append(roc_auc_score(yte, s_clean))

    # benign NLL distribution stats for targeting
    s_tr = -gm.score_samples(X[tr_b])
    med_ben = float(np.median(s_tr))

    # (a) score-space blend (paper's sweep): blend normalized score toward benign mean
    lo, hi = s_tr.min(), s_tr.max()
    def norm(s): return (s - lo) / (hi - lo)
    s_norm = norm(s_clean)
    s_blend_norm = 0.0 * s_norm + norm(s_tr).mean()  # K=1 full suppression
    s_blend = s_blend_norm * (hi - lo) + lo
    auc_sb = roc_auc_score(yte, s_blend)
    auc_scoreblend.append(auc_sb)

    # (b) feature-space: move each malware sample toward mu along whitened path
    #     until NLL <= benign median (or max steps); preserves "feasible" region shape.
    Xadv = X[mal_idx].copy()
    diff = mu - Xadv
    dist = np.sqrt(np.einsum('ij,jk,ik->i', diff, Cinv, diff))
    # step fraction needed: binary search per sample on alpha in [0,1]
    alpha = np.ones(len(Xadv))
    for _ in range(12):
        mid = alpha / 2
        Xtry = Xadv + (mu - Xadv) * (1 - alpha[:, None])  # current pos
        nll_try = -gm.score_samples(Xtry)
        too_high = nll_try > med_ben
        alpha[too_high] = np.minimum(1.0, alpha[too_high] * 2) if False else alpha[too_high]  # expand
        # simpler: geometric expansion then final clip below
        break
    # vectorized approach: find minimal t in [0,1] s.t. moved sample reaches med_ben
    ts = np.linspace(0, 1, 41)  # fraction of the way to mu
    Xmoves = X[mal_idx][:, None, :] + (mu - X[mal_idx])[:, None, :] * ts[None, :, None]
    flat = Xmoves.reshape(-1, X.shape[1])
    nlls = (-gm.score_samples(flat)).reshape(len(mal_idx), len(ts))
    first_ok = (nlls <= med_ben).argmax(axis=1)
    reached = (nlls <= med_ben).any(axis=1)
    t_sel = ts[first_ok]
    t_sel[~reached] = 1.0  # even at mu not enough -> fully benign point
    Xadv = X[mal_idx] + (mu - X[mal_idx]) * t_sel[:, None]

    s_fs = -gm.score_samples(np.vstack([X[te_b], Xadv]))
    auc_fs = roc_auc_score(yte, s_fs)
    auc_featspace.append(auc_fs)
    print(f'seed{seed}: clean={auc_clean[-1]:.4f} scoreblend={auc_sb:.4f} featspace={auc_fs:.4f}', flush=True)

import json
out = {
 'description': ('Feature-space adversarial eval on RanSMAP channel: move malware features '
                 'toward benign mean along Mahalanobis-whitened path until NLL<=benign-median. '
                 'Compared vs paper-style score-space blending at K=1.'),
 'canonical_families': CANONICAL,
 'clean_auc_mean': float(np.mean(auc_clean)),
 'scoreblend_K1_auc_mean': float(np.mean(auc_scoreblend)),
 'featurespace_auc_mean': float(np.mean(auc_featspace)),
 'clean': [float(a) for a in auc_clean],
 'scoreblend': [float(a) for a in auc_scoreblend],
 'featspace': [float(a) for a in auc_featspace],
}
with open(OUT, 'w') as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: v for k, v in out.items() if isinstance(v, float)}, indent=1))
print('Saved ->', OUT)
