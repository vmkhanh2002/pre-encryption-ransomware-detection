"""Exact TPR at low FPR by threshold counting instead of ROC interpolation.

At the FPR values a finite test set can actually resolve (multiples of
1/n_benign), count true and false positives at the k-th highest benign
score rather than interpolating the ROC curve. Also reports the smallest
FPR each test set can support (paper, Section 4.2, 0.1%-FPR paragraph).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RESULTS_DIR, SEEDS, K_MLRAN, COV_MLRAN, K_RANSMAP, COV_RANSMAP, REG_COVAR, N_INIT  # noqa: E402
import os, sys, hashlib, json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from utils import load_mlran, load_ransmap

OUT = RESULTS_DIR / 'exact_fpr_counts.json'

results = {'description': ('Exact TPR@k-th-benign-percentile counting. The smallest '
                           'empirically supportable FPR is 1/n_ben_test; any smaller '
                           'target requires extrapolation. Reported per channel.'),
           'channels': {}}

# ---------- MLRan ----------
dfm, fm = load_mlran()
Xm = dfm[fm].to_numpy(np.float32)
ym = dfm['label'].to_numpy()
groups = dfm[fm].astype(str).agg('|'.join, axis=1).apply(
    lambda s: hashlib.md5(s.encode()).hexdigest())

mlran_res = []
for seed in SEEDS:
    tr_g, _ = train_test_split(groups.unique(), test_size=0.2, random_state=seed)
    mtr = groups.isin(set(tr_g)).to_numpy()
    gm = GaussianMixture(K_MLRAN, covariance_type=COV_MLRAN, reg_covar=REG_COVAR,
                         n_init=N_INIT, random_state=seed).fit(Xm[mtr & (ym == 0)])
    s_all = -gm.score_samples(Xm[~mtr])
    yte = ym[~mtr]
    s_ben = s_all[yte == 0]
    s_mal = s_all[yte == 1]
    nb = len(s_ben)
    # thresholds: k-th largest benign score for k = ceil(q*nb), q in {0.001, 0.01, 0.05}
    row = {'n_ben_test': int(nb), 'n_mal_test': int(len(s_mal))}
    for q in (0.001, 0.01, 0.05):
        k = max(1, int(np.ceil(q * nb)))
        thr = np.sort(s_ben)[-k]          # k-th largest benign score
        fpr_exact = float((s_ben >= thr).mean())
        tpr_exact = float((s_mal >= thr).mean())
        row[f'q{q}'] = {'k_threshold': int(k), 'exact_fpr': round(fpr_exact, 5),
                        'exact_tpr': round(tpr_exact, 4)}
    mlran_res.append(row)

results['channels']['mlran'] = {
    'per_seed': mlran_res,
    'min_supportable_fpr': round(float(np.mean([1/r['n_ben_test'] for r in mlran_res])), 5),
}

# ---------- RanSMAP ----------
dfr, fr = load_ransmap()
Xr = dfr[fr].to_numpy(np.float64)
is_ben = dfr['label'].to_numpy() == 0

ransmap_res = []
for seed in SEEDS:
    idx_ben = np.where(is_ben)[0]
    tr_b, te_b = train_test_split(idx_ben, test_size=0.2, random_state=seed)
    mal_idx = np.where(~is_ben)[0]
    Xre = np.vstack([Xr[te_b], Xr[mal_idx]])
    yre = np.concatenate([np.zeros(len(te_b)), np.ones(len(mal_idx))])
    gm = GaussianMixture(K_RANSMAP, covariance_type=COV_RANSMAP, reg_covar=REG_COVAR,
                         n_init=N_INIT, random_state=seed).fit(Xr[tr_b])
    s_all = -gm.score_samples(Xre)
    s_ben = s_all[yre == 0]; s_mal = s_all[yre == 1]
    nb = len(s_ben)
    row = {'n_ben_test': int(nb), 'n_mal_test': int(len(s_mal))}
    for q in (0.001, 0.01, 0.05):
        k = max(1, int(np.ceil(q * nb)))
        thr = np.sort(s_ben)[-k]
        row[f'q{q}'] = {'k_threshold': int(k),
                        'exact_fpr': round(float((s_ben >= thr).mean()), 5),
                        'exact_tpr': round(float((s_mal >= thr).mean()), 4)}
    ransmap_res.append(row)

results['channels']['ransmap'] = {
    'per_seed': ransmap_res,
    'min_supportable_fpr': round(float(np.mean([1/r['n_ben_test'] for r in ransmap_res])), 5),
}

with open(OUT, 'w') as f:
    json.dump(results, f, indent=1)

# summary
import numpy as np2
for ch in ('mlran', 'ransmap'):
    per = results['channels'][ch]['per_seed']
    for q in ('q0.001', 'q0.01', 'q0.05'):
        tprs = [p[q]['exact_tpr'] for p in per]
        fprs = [p[q]['exact_fpr'] for p in per]
        print(f'{ch} {q}: mean TPR={np.mean(tprs):.4f} mean FPR={np.mean(fprs):.4f}')
print('Saved ->', OUT)
