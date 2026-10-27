"""Feature-freeze test for the white-box nearest-mode attack.

Question: does evasion require moving the bulk write-throughput features
(w_bytes_per_s, w_total_bytes, w_io_count, w_io_per_s, w_sequential_frac),
or can the attacker evade while keeping them at the ransomware's own values
and moving only the other 29 features?

Method: the same GMM fit, nearest-benign-mode target and per-sample alpha
bisection as whitebox_mode_attack.py, with the interpolation masked so the
frozen dimensions never move. Success rate and post-attack AUC are compared
between the unconstrained and the write-frozen attack (10 seeds, 5 canonical
families). whitebox_feature_freeze_control.py provides the randomized
control needed to interpret the result (paper, Section 4.2).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RESULTS_DIR, SEEDS, K_RANSMAP, COV_RANSMAP, REG_COVAR, N_INIT  # noqa: E402
import os, sys, json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import load_ransmap

OUT = RESULTS_DIR / 'whitebox_feature_freeze.json'
df, feat_cols = load_ransmap()
X = df[feat_cols].to_numpy(dtype=np.float64)
fam = df['family'].to_numpy()
is_benign = df['label'].to_numpy() == 0
CANONICAL = ['Conti', 'Ryuk', 'LockBit', 'REvil', 'WannaCry']

WRITE_SIG_FEATS = ['w_bytes_per_s', 'w_total_bytes', 'w_io_count',
                   'w_io_per_s', 'w_sequential_frac']
write_sig_idx = [feat_cols.index(f) for f in WRITE_SIG_FEATS]
mask_frozen = np.ones(len(feat_cols))
mask_frozen[write_sig_idx] = 0.0   # 0 = frozen (write-signature group), 1 = movable

def nll(gm, Xq):
    return -gm.score_samples(Xq)

def solve_alpha_masked(gm, X0, T, tau, mask, max_a=1.0):
    """Per-sample smallest a with NLL(x + a*mask*(T-x)) <= tau."""
    def f(a):
        aa = np.asarray(a)
        if aa.ndim == 0:
            return nll(gm, X0 + float(aa) * mask[None, :] * (T - X0))
        return nll(gm, X0 + aa[:, None] * mask[None, :] * (T - X0))
    n = len(X0)
    alphas = np.zeros(n)
    inside = f(0.0) <= tau
    reachable = f(max_a) <= tau
    todo = (~inside) & reachable
    lo = np.zeros(n); hi = np.full(n, max_a)
    for _ in range(30):
        mid = (lo + hi) / 2
        ok = f(mid) <= tau
        hi = np.where(ok & todo, mid, hi)
        lo = np.where(~ok & todo, mid, lo)
    alphas[todo] = hi[todo]
    success = inside | reachable
    return alphas, success

res = {'description': __doc__.split('\n')[0], 'frozen_features': WRITE_SIG_FEATS,
       'per_seed': {}}
agg = {k: [] for k in ['clean_auc', 'unconstrained_auc', 'unconstrained_success',
                       'frozen_auc', 'frozen_success']}

for seed in SEEDS:
    idx_b = np.where(is_benign)[0]
    tr_b, te_b = train_test_split(idx_b, test_size=0.2, random_state=seed)
    gm = GaussianMixture(K_RANSMAP, covariance_type=COV_RANSMAP, reg_covar=REG_COVAR,
                         n_init=N_INIT, random_state=seed).fit(X[tr_b])
    mus = gm.means_
    covs = gm.covariances_
    s_tr = nll(gm, X[tr_b])
    tau = float(np.median(s_tr))

    mal_idx = np.array([i for i in np.where(~is_benign)[0] if fam[i] in CANONICAL])
    Xte = np.vstack([X[te_b], X[mal_idx]])
    yte = np.concatenate([np.zeros(len(te_b)), np.ones(len(mal_idx))])
    Xmal = X[mal_idx]
    a_clean = roc_auc_score(yte, nll(gm, Xte))

    def mahal_to_centers(Xq):
        D = np.empty((len(Xq), len(mus)))
        for k in range(len(mus)):
            diff = Xq - mus[k]
            D[:, k] = np.einsum('ij,jk,ik->i', diff, covs[k], diff)
        return D
    kstar = mahal_to_centers(Xmal).argmin(axis=1)
    T_mode = mus[kstar]

    mask_all = np.ones(len(feat_cols))
    alphas_u, succ_u = solve_alpha_masked(gm, Xmal, T_mode, tau, mask_all)
    Xadv_u = Xmal + alphas_u[:, None] * mask_all[None, :] * (T_mode - Xmal)
    auc_u = roc_auc_score(yte, nll(gm, np.vstack([X[te_b], Xadv_u])))

    alphas_f, succ_f = solve_alpha_masked(gm, Xmal, T_mode, tau, mask_frozen)
    Xadv_f = Xmal + alphas_f[:, None] * mask_frozen[None, :] * (T_mode - Xmal)
    auc_f = roc_auc_score(yte, nll(gm, np.vstack([X[te_b], Xadv_f])))

    out = {'clean_auc': float(a_clean),
           'unconstrained': {'success_rate': float(succ_u.mean()), 'post_attack_auc': float(auc_u)},
           'write_sig_frozen': {'success_rate': float(succ_f.mean()), 'post_attack_auc': float(auc_f)}}
    res['per_seed'][f'seed{seed}'] = out
    agg['clean_auc'].append(out['clean_auc'])
    agg['unconstrained_auc'].append(out['unconstrained']['post_attack_auc'])
    agg['unconstrained_success'].append(out['unconstrained']['success_rate'])
    agg['frozen_auc'].append(out['write_sig_frozen']['post_attack_auc'])
    agg['frozen_success'].append(out['write_sig_frozen']['success_rate'])
    print(f"seed {seed}: unconstrained succ={out['unconstrained']['success_rate']:.3f} auc={out['unconstrained']['post_attack_auc']:.3f} | "
          f"frozen succ={out['write_sig_frozen']['success_rate']:.3f} auc={out['write_sig_frozen']['post_attack_auc']:.3f}", flush=True)

res['summary'] = {
    'clean_auc_mean': float(np.mean(agg['clean_auc'])),
    'unconstrained_success_mean': float(np.mean(agg['unconstrained_success'])),
    'unconstrained_auc_mean': float(np.mean(agg['unconstrained_auc'])),
    'write_sig_frozen_success_mean': float(np.mean(agg['frozen_success'])),
    'write_sig_frozen_auc_mean': float(np.mean(agg['frozen_auc'])),
}
with open(OUT, 'w') as f:
    json.dump(res, f, indent=1)
print(json.dumps(res['summary'], indent=1))
print('saved ->', OUT)
