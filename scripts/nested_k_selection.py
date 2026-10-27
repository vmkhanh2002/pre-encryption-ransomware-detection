"""Nested K selection on the training split only.

Selects the per-channel GMM component count by BIC using the training split
alone, with the deployment reg_covar, to check how the K values used in the
paper compare with a nested, leakage-free selection. High K on MLRan's
binary data can collapse groups; such fits are caught and recorded as
failures. Features are standardised before BIC.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RESULTS_DIR, SEEDS, REG_COVAR  # noqa: E402
import os, sys, hashlib, json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from utils import load_mlran, load_ransmap

N_INIT_SWEEP = 3  # EM restarts per candidate K in the nested BIC sweep

OUT = RESULTS_DIR / 'nested_k_selection.json'

results = {'description': (f'Nested BIC K-selection on TRAIN split only, deployment reg_covar={REG_COVAR:g}. '
                           'Paper chose K via full-data BIC: MLRan K=3 (elbow), RanSMAP K=5 (BIC min).'),
           'mlran': {}, 'ransmap': {}}

# ---- MLRan ----
df, feat_cols = load_mlran()
X_all = df[feat_cols].to_numpy(dtype=np.float32)
group_ids = df[feat_cols].astype(str).agg('|'.join, axis=1).apply(
    lambda s: hashlib.md5(s.encode()).hexdigest())
uniq_groups = group_ids.unique()

mlran_ks = []
for seed in SEEDS:
    tr_groups, _ = train_test_split(uniq_groups, test_size=0.2, random_state=seed)
    Xtr = X_all[group_ids.isin(set(tr_groups)).to_numpy()]
    bics = {}
    for K in range(1, 8):
        try:
            gm = GaussianMixture(n_components=K, covariance_type='diag', reg_covar=REG_COVAR,
                                 n_init=N_INIT_SWEEP, max_iter=300, random_state=seed).fit(Xtr)
            bics[K] = float(gm.bic(Xtr))
        except ValueError:
            bics[K] = None  # ill-defined at this K on this split
    valid = {k: v for k, v in bics.items() if v is not None}
    best_K = min(valid, key=valid.get) if valid else None
    results['mlran'][f'seed{seed}'] = {'bic': bics, 'best_K': best_K}
    mlran_ks.append(best_K)
    print(f'MLRan seed{seed}: best_K={best_K}', flush=True)

results['mlran']['summary'] = {
    'paper_K': 3, 'chosen_ks': [k if k is None else int(k) for k in mlran_ks],
    'modal_K': max([k for k in mlran_ks if k], key=lambda k: mlran_ks.count(k)) if any(mlran_ks) else None,
    'frac_matching_paper': round(sum(k == 3 for k in mlran_ks) / len(mlran_ks), 2)}

# ---- RanSMAP ----
df2, feat2 = load_ransmap()
X2 = df2[feat2].to_numpy(dtype=np.float64)
is_benign = df2['label'].to_numpy() == 0
Xb = X2[is_benign]

ransmap_ks = []
for seed in SEEDS:
    Xtr, _ = train_test_split(Xb, test_size=0.2, random_state=seed)
    bics = {}
    for K in range(1, 8):
        try:
            gm = GaussianMixture(n_components=K, covariance_type='full', reg_covar=REG_COVAR,
                                 n_init=N_INIT_SWEEP, max_iter=300, random_state=seed).fit(Xtr)
            bics[K] = float(gm.bic(Xtr))
        except ValueError:
            bics[K] = None
    valid = {k: v for k, v in bics.items() if v is not None}
    best_K = min(valid, key=valid.get) if valid else 5
    results['ransmap'][f'seed{seed}'] = {'bic': bics, 'best_K': int(best_K)}
    ransmap_ks.append(int(best_K))
    print(f'RanSMAP seed{seed}: best_K={best_K}', flush=True)

results['ransmap']['summary'] = {
    'paper_K': 5, 'chosen_ks': ransmap_ks,
    'modal_K': max(set(ransmap_ks), key=ransmap_ks.count),
    'frac_matching_paper': round(sum(k == 5 for k in ransmap_ks) / len(ransmap_ks), 2)}

with open(OUT, 'w') as f:
    json.dump(results, f, indent=1)
print('MLRan summary:', results['mlran']['summary'])
print('RanSMAP summary:', results['ransmap']['summary'])
print('Saved ->', OUT)
