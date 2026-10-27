"""Bernoulli mixture vs. diagonal GMM on MLRan's binary features.

A Bernoulli mixture is the natural density model for 0/1 indicators, so a
diagonal Gaussian is formally misspecified here. This script fits both
(K = K_MLRAN, benign-only, group-based split, 10 seeds) and compares
detection AUC, to check whether the misspecification costs accuracy
(paper, Section 5.2).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import COV_MLRAN, K_MLRAN, N_INIT, REG_COVAR, RESULTS_DIR, SEEDS, TEST_SIZE  # noqa: E402
import os, sys, hashlib, json
import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import load_mlran

OUT = RESULTS_DIR / 'bernoulli_vs_gmm.json'

df, feat_cols = load_mlran()
X = df[feat_cols].to_numpy(dtype=np.float32)
y = df['label'].to_numpy()
group_ids = df[feat_cols].astype(str).agg('|'.join, axis=1).apply(
    lambda s: hashlib.md5(s.encode()).hexdigest())
uniq_groups = group_ids.unique()

res = {'gmm_diag': [], 'bernoulli': []}

for seed in SEEDS:
    tr_g, te_g = train_test_split(uniq_groups, test_size=TEST_SIZE, random_state=seed)
    mtr = group_ids.isin(set(tr_g)).to_numpy()
    Xtr, ytr = X[mtr], y[mtr]
    Xte, yte = X[~mtr], y[~mtr]

    # (a) GMM diag (paper model), benign-only
    gm = GaussianMixture(K_MLRAN, covariance_type=COV_MLRAN, reg_covar=REG_COVAR,
                         n_init=N_INIT, random_state=seed).fit(Xtr[ytr == 0])
    auc_gmm = roc_auc_score(yte, -gm.score_samples(Xte))

    # (b) Bernoulli mixture K=3 via EM, benign-only
    Xb = Xtr[ytr == 0].astype(np.float32)
    K = K_MLRAN
    rng = np.random.RandomState(seed)
    resp = rng.dirichlet(np.ones(K), size=len(Xb))
    pi = resp.mean(axis=0)
    theta = np.clip((resp.T @ Xb) / resp.sum(axis=0)[:, None], 1e-6, 1 - 1e-6)
    for _ in range(50):
        logp = np.log(theta)[None, :, :] * Xb[:, None, :] + \
               np.log(1 - theta)[None, :, :] * (1 - Xb[:, None, :])
        logresp = np.log(pi + 1e-12)[None, :] + logp.sum(axis=2)
        logresp -= logresp.max(axis=1, keepdims=True)
        resp = np.exp(logresp); resp /= resp.sum(axis=1, keepdims=True)
        Nk = resp.sum(axis=0) + 1e-12
        pi = Nk / len(Xb)
        theta = np.clip((resp.T @ Xb) / Nk[:, None], 1e-6, 1 - 1e-6)

    Xte32 = Xte.astype(np.float32)
    logp_te = np.log(theta)[None, :, :] * Xte32[:, None, :] + \
              np.log(1 - theta)[None, :, :] * (1 - Xte32[:, None, :])
    loglik = logp_te.sum(axis=2) + np.log(pi + 1e-12)[None, :]
    nll = -logsumexp(loglik, axis=1)
    auc_ber = roc_auc_score(yte, nll)

    res['gmm_diag'].append(float(auc_gmm))
    res['bernoulli'].append(float(auc_ber))
    print(f'seed{seed}: GMM={auc_gmm:.4f} Bernoulli={auc_ber:.4f}', flush=True)

summary = {
    'description': 'Benign-only AUC on group-based split: diag GMM(K=3) vs Bernoulli mixture(K=3)',
    'gmm_diag_mean': float(np.mean(res['gmm_diag'])),
    'gmm_diag_std': float(np.std(res['gmm_diag'], ddof=1)),
    'bernoulli_mean': float(np.mean(res['bernoulli'])),
    'bernoulli_std': float(np.std(res['bernoulli'], ddof=1)),
}
with open(OUT, 'w') as f:
    json.dump({**summary, **res}, f, indent=1)
print(json.dumps(summary, indent=1))
print('Saved ->', OUT)
