"""Zero-knowledge throttling (rate-limiting) evasion sweep.

The threat model lets the attacker encrypt intermittently (paper, Section
5.1). Unlike the Level-2/3 attacks, throttling needs no knowledge of the
detector, the GMM, or any benign statistic: the same encryption I/O is
spread over more wall-clock time (e.g. sleeping between chunks).

Physical model: total work is fixed (the same files must be encrypted, so
total bytes, I/O count, size distribution and sector footprint cannot
shrink), but duration can grow by a slowdown factor k. This divides the rate
features (io_per_s, bytes_per_s) by k and multiplies duration_s by k, for
reads and writes; all other features keep the malware's own values. It
corresponds to the post-encryption delay knob of Hirano and Kobayashi
(IEEE CSR 2025) in feature space (paper, Section 4.2).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RESULTS_DIR, SEEDS, K_RANSMAP, COV_RANSMAP, REG_COVAR, N_INIT  # noqa: E402
import os, json
import sys
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import load_ransmap

OUT = RESULTS_DIR / 'throttling_evasion.json'
CANONICAL = ['Conti', 'Ryuk', 'LockBit', 'REvil', 'WannaCry']
KS = [1, 2, 5, 10, 20, 50, 100]

df, feat_cols = load_ransmap()
X = df[feat_cols].to_numpy(dtype=np.float64)
fam = df['family'].to_numpy()
is_benign = df['label'].to_numpy() == 0

RATE_PAIRS = [('r_io_per_s', 'r_duration_s'), ('w_io_per_s', 'w_duration_s'),
              ('r_bytes_per_s', 'r_duration_s'), ('w_bytes_per_s', 'w_duration_s')]
idx_r_io_per_s = feat_cols.index('r_io_per_s')
idx_w_io_per_s = feat_cols.index('w_io_per_s')
idx_r_bytes_per_s = feat_cols.index('r_bytes_per_s')
idx_w_bytes_per_s = feat_cols.index('w_bytes_per_s')
idx_r_duration_s = feat_cols.index('r_duration_s')
idx_w_duration_s = feat_cols.index('w_duration_s')

def throttle(Xmal, k):
    """Slow down by factor k: duration *= k, rate features /= k. Totals/shape untouched."""
    Xt = Xmal.copy()
    Xt[:, idx_r_duration_s] *= k
    Xt[:, idx_w_duration_s] *= k
    Xt[:, idx_r_io_per_s] /= k
    Xt[:, idx_w_io_per_s] /= k
    Xt[:, idx_r_bytes_per_s] /= k
    Xt[:, idx_w_bytes_per_s] /= k
    return Xt

def nll(gm, Xq):
    return -gm.score_samples(Xq)

res = {'description': __doc__.split('\n')[0],
       'note': ('Zero-model-knowledge attack: only inflates duration and deflates '
                'the two rate features per direction; w_total_bytes, w_io_count and '
                'w_sequential_frac (three of the five necessity-ablation-frozen '
                'features) are left untouched.'),
       'ks': KS, 'per_seed': {}}
agg = {k: {'auc': [], 'evasion_frac': [], 'detected_frac_at_tau': []} for k in KS}

for seed in SEEDS:
    idx_b = np.where(is_benign)[0]
    tr_b, te_b = train_test_split(idx_b, test_size=0.2, random_state=seed)
    gm = GaussianMixture(K_RANSMAP, covariance_type=COV_RANSMAP, reg_covar=REG_COVAR,
                         n_init=N_INIT, random_state=seed).fit(X[tr_b])
    s_tr = nll(gm, X[tr_b])
    tau = float(np.median(s_tr))

    mal_idx = np.array([i for i in np.where(~is_benign)[0] if fam[i] in CANONICAL])
    Xte_b = X[te_b]
    Xmal = X[mal_idx]
    yte = np.concatenate([np.zeros(len(te_b)), np.ones(len(mal_idx))])

    seed_out = {}
    for k in KS:
        Xadv = throttle(Xmal, k) if k > 1 else Xmal
        s_adv = nll(gm, Xadv)
        auc = roc_auc_score(yte, np.concatenate([nll(gm, Xte_b), s_adv]))
        evasion_frac = float(np.mean(s_adv < tau))
        detected_frac = float(np.mean(s_adv >= tau))
        seed_out[f'k{k}'] = {'auc': float(auc), 'evasion_frac': evasion_frac,
                             'detected_frac_at_tau': detected_frac}
        agg[k]['auc'].append(auc)
        agg[k]['evasion_frac'].append(evasion_frac)
        agg[k]['detected_frac_at_tau'].append(detected_frac)
    res['per_seed'][f'seed{seed}'] = seed_out
    print(f"seed {seed}: " + " | ".join(
        f"k={k} auc={seed_out[f'k{k}']['auc']:.3f} evade={seed_out[f'k{k}']['evasion_frac']:.2f}"
        for k in KS), flush=True)

res['summary'] = {
    f'k{k}': {'auc_mean': float(np.mean(agg[k]['auc'])),
              'evasion_frac_mean': float(np.mean(agg[k]['evasion_frac'])),
              'detected_frac_at_tau_mean': float(np.mean(agg[k]['detected_frac_at_tau']))}
    for k in KS
}
with open(OUT, 'w') as f:
    json.dump(res, f, indent=1)
print(json.dumps(res['summary'], indent=1))
print('saved ->', OUT)
