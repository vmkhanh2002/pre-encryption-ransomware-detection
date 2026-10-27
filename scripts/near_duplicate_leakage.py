"""Near-duplicate leakage check and hardened group split for MLRan.

Exact-hash grouping misses near-duplicates (vectors differing in 1-3 bits).
This script:
1. counts cross train/test pairs within a small Hamming radius under the
   exact-hash split (leakage estimate);
2. builds hardened groups by union-find over pairs within radius r
   (brute force over the 1,818 unique vectors, packed uint8 XOR + popcount);
3. reruns the benign-only GMM AUC with hardened vs. exact-hash groups.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RESULTS_DIR, K_MLRAN, COV_MLRAN, REG_COVAR, N_INIT, SEEDS  # noqa: E402
import os, sys, hashlib, json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import load_mlran

OUT = RESULTS_DIR / 'near_duplicate_leakage.json'

df, feat_cols = load_mlran()
X = df[feat_cols].to_numpy(np.uint8)
y = df['label'].to_numpy()
N = len(X)

# unique vectors + mapping row->unique id
uniq, inv = np.unique(X, axis=0, return_inverse=True)
U = len(uniq)
print(f'rows={N} unique_vectors={U}', flush=True)

# pack bits: 482 bits -> 61 bytes
packed = np.packbits(uniq, axis=1)  # (U, 61)

def hamming_matrix_chunked(packed, chunk=200):
    U = len(packed)
    D = np.zeros((U, U), dtype=np.uint16)
    for i in range(0, U, chunk):
        xi = packed[i:i+chunk]
        for j in range(i, U, chunk):
            xj = packed[j:j+chunk]
            xor = xi[:, None, :] ^ xj[None, :, :]
            d = np.unpackbits(xor, axis=2).sum(axis=2)
            D[i:i+len(xi), j:j+len(xj)] = d
            if i != j:
                D[j:j+len(xj), i:i+len(xi)] = d.T
    return D

print('computing pairwise Hamming on unique vectors...', flush=True)
D = hamming_matrix_chunked(packed)

class UF:
    def __init__(s, n): s.p = list(range(n))
    def find(s, a):
        while s.p[a] != a:
            s.p[a] = s.p[s.p[a]]; a = s.p[a]
        return a
    def union(s, a, b): s.p[s.find(a)] = s.find(b)

def grouped_split(hard_groups, seed):
    rng = np.random.RandomState(seed)
    ug = np.unique(hard_groups)
    tr, te = train_test_split(ug, test_size=0.2, random_state=seed)
    return np.isin(hard_groups, tr)

res = {'description': ('Near-duplicate leakage analysis. Radius r means two unique '
                       'vectors differing in <=r of 482 bits are treated as same group.'),
       'radii': {}}

for r in [0, 1, 2, 3, 5]:
    uf = UF(U)
    if r > 0:
        ii, jj = np.where((D <= r) & (D > 0))
        for a, b in zip(ii, jj):
            uf.union(int(a), int(b))
        n_pairs = len(ii) // 2
    else:
        n_pairs = 0
    hard = np.array([uf.find(inv[i]) for i in range(N)])
    n_hard = len(np.unique(hard))

    # leakage metric: fraction of test rows whose vector is within r of a train row
    leak_fracs = []
    aucs_exact, aucs_hard = [], []
    for seed in SEEDS:
        # exact-hash split
        ug_ex = np.unique(inv)
        tr_e, te_e = train_test_split(ug_ex, test_size=0.2, random_state=seed)
        mtr_e = np.isin(inv, tr_e); mte_e = ~mtr_e
        # leakage: min distance from each test unique-vector to any train unique-vector
        te_u = np.unique(inv[mte_e]); tr_u = np.unique(inv[mtr_e])
        mind = D[te_u][:, tr_u].min(axis=1)
        leak_fracs.append(float((mind <= r).mean()))

        # AUC with hardened groups
        mtr_h = grouped_split(hard, seed)
        Xtr_ben = X[mtr_h & (y == 0)]
        gm = GaussianMixture(K_MLRAN, covariance_type=COV_MLRAN, reg_covar=REG_COVAR,
                             n_init=N_INIT, random_state=seed).fit(Xtr_ben)
        auc = roc_auc_score(y[~mtr_h], -gm.score_samples(X[~mtr_h]))
        aucs_hard.append(auc)

        # baseline AUC with exact groups (same seeds)
        Xtr_b = X[mtr_e & (y == 0)]
        gm2 = GaussianMixture(K_MLRAN, covariance_type=COV_MLRAN, reg_covar=REG_COVAR,
                              n_init=N_INIT, random_state=seed).fit(Xtr_b)
        auc2 = roc_auc_score(y[mte_e], -gm2.score_samples(X[mte_e]))
        aucs_exact.append(auc2)

    res['radii'][r] = {
        'unique_groups': int(n_hard),
        'merge_pairs_at_r': n_pairs,
        'mean_leak_frac_train_prox': round(float(np.mean(leak_fracs)), 4),
        'auc_exact_mean': round(float(np.mean(aucs_exact)), 4),
        'auc_hardened_mean': round(float(np.mean(aucs_hard)), 4),
        'auc_delta': round(float(np.mean(aucs_hard) - np.mean(aucs_exact)), 4),
    }
    print(f'r={r}:', res['radii'][r], flush=True)

with open(OUT, 'w') as f:
    json.dump(res, f, indent=1)
print('Saved ->', OUT)
