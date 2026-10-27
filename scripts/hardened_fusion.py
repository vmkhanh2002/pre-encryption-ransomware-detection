"""Fused AUC with a near-duplicate-hardened MLRan split.

Compares the fused AUC under the exact-hash group split used in the paper
with an r=1 Hamming-hardened grouping (vectors within one bit merged into
the same group), with the same protocol (5-family LOFO, 10 seeds,
benign-only training) and the learned meta-GMM fusion of learned_fusion.py.
Backs the paper's statement that near-duplicate hardening leaves the fused
result unchanged (Section 5.2).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RESULTS_DIR, SEEDS, K_MLRAN, COV_MLRAN, K_RANSMAP, COV_RANSMAP, REG_COVAR, N_INIT, META_REG_COVAR, selected_meta_k  # noqa: E402
import os, sys, hashlib, json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import load_mlran, load_ransmap

OUT = RESULTS_DIR / 'hardened_fusion.json'

CANON_R = ['Conti', 'Ryuk', 'LockBit', 'REvil', 'WannaCry']
META_K = selected_meta_k()

dfm, fm = load_mlran()
Xm = dfm[fm].to_numpy(np.uint8)
ym = dfm['label'].to_numpy()
fam_m = np.array([str(f).lower() for f in dfm['family'].to_numpy()])

# ---- build hardened groups at r=1 over unique vectors ----
uniq, inv = np.unique(Xm, axis=0, return_inverse=True)
packed = np.packbits(uniq, axis=1)
U = len(packed)

# find pairs within hamming distance 1: for each bit position b, group vectors
# by their bits with position b zeroed; two vectors differ only at b iff they
# collide there. O(U * 482) hashing instead of O(U^2).
hard_id = list(range(U))
parent = list(range(U))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb

for b in range(482):
    byte_i, bit_i = b // 8, 7 - (b % 8)
    buckets = {}
    for v in range(U):
        key = packed[v].copy()
        key[byte_i] = key[byte_i] & (255 - (1 << bit_i))
        k = key.tobytes()
        if k in buckets:
            union(v, buckets[k])
        else:
            buckets[k] = v

hard_root = np.array([find(inv[i]) for i in range(len(Xm))])
n_hard_groups = len(set(hard_root.tolist()))
print(f'exact groups={len(np.unique(inv))}, r1-hardened groups={n_hard_groups}', flush=True)

dfr, fr = load_ransmap()
Xr = dfr[fr].to_numpy(np.float64)
is_ben = dfr['label'].to_numpy() == 0
famr = dfr['family'].to_numpy()


def split_half(arr, rng):
    idx = rng.permutation(len(arr))
    h = len(arr) // 2
    return arr[idx[:h]], arr[idx[h:]]


def pair(a, b, rng, n):
    ia = rng.choice(len(a), n, replace=(len(a) < n))
    ib = rng.choice(len(b), n, replace=(len(b) < n))
    return a[ia], b[ib]


def run_fusion(split_mode, seed):
    """Returns per-family fused B_learned AUC dict."""
    rng_split, rng_meta, rng_pair = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))

    if split_mode == 'exact':
        ug = np.unique(inv)
        tr_g, _ = train_test_split(ug, test_size=0.2, random_state=seed)
        mtr = np.isin(inv, tr_g)
    else:
        ug = np.unique(hard_root)
        tr_g, _ = train_test_split(ug, test_size=0.2, random_state=seed)
        mtr = np.isin(hard_root, tr_g)

    gm_m = GaussianMixture(K_MLRAN, covariance_type=COV_MLRAN, reg_covar=REG_COVAR,
                           n_init=N_INIT, random_state=seed).fit(Xm[mtr & (ym == 0)].astype(np.float64))
    te_mask = ~mtr
    sm_all = -gm_m.score_samples(Xm[te_mask].astype(np.float64))
    fme = fam_m[te_mask]; yme = ym[te_mask]

    idx_ben = np.where(is_ben)[0]
    tr_b, te_b = train_test_split(idx_ben, test_size=0.2, random_state=seed)
    gm_r = GaussianMixture(K_RANSMAP, covariance_type=COV_RANSMAP, reg_covar=REG_COVAR,
                           n_init=N_INIT, random_state=seed).fit(Xr[tr_b])
    mal_idx = [i for i in np.where(~is_ben)[0] if famr[i] in CANON_R]
    Xre = np.vstack([Xr[te_b], Xr[mal_idx]])
    sr_all = -gm_r.score_samples(Xre)
    off_r = len(te_b)

    s_ben_m_tr = -gm_m.score_samples(Xm[mtr & (ym == 0)].astype(np.float64))
    s_ben_r_tr = -gm_r.score_samples(Xr[tr_b])
    lo_m, hi_m = s_ben_m_tr.min(), s_ben_m_tr.max()
    lo_r, hi_r = s_ben_r_tr.min(), s_ben_r_tr.max()
    nm = lambda s: np.clip((s - lo_m) / max(hi_m - lo_m, 1e-9), 0, None)
    nr_ = lambda s: np.clip((s - lo_r) / max(hi_r - lo_r, 1e-9), 0, None)

    # leak-free calib/eval split of benign TEST scores, per channel, then a
    # meta-GMM fit ONLY on the calibration half (matches learned_fusion.py)
    sm_ben_te_n = nm(sm_all[yme == 0])
    sr_ben_te_n = nr_(sr_all[:off_r])
    sm_ben_calib, sm_ben_eval = split_half(sm_ben_te_n, rng_split)
    sr_ben_calib, sr_ben_eval = split_half(sr_ben_te_n, rng_split)
    n_calib = min(len(sm_ben_calib), len(sr_ben_calib))
    pa_c, pb_c = pair(sm_ben_calib, sr_ben_calib, rng_meta, max(n_calib, 200))
    meta = GaussianMixture(META_K, covariance_type='full', reg_covar=META_REG_COVAR,
                           n_init=N_INIT, random_state=seed).fit(np.column_stack([pa_c, pb_c]))
    n_eval = min(len(sm_ben_eval), len(sr_ben_eval))

    out = {}
    for f in CANON_R:
        mi = np.where((fme == f.lower()) & (yme == 1))[0]
        ri_local = [j for j, i in enumerate(mal_idx) if famr[i] == f]
        n = min(len(mi), len(ri_local))
        if n == 0:
            continue
        sm_mal_n = nm(sm_all[mi])
        sr_mal_n = nr_(sr_all[off_r + np.array(ri_local)])
        pa_m, pb_m = pair(sm_mal_n, sr_mal_n, rng_pair, n)
        pa_be, pb_be = pair(sm_ben_eval, sr_ben_eval, rng_pair, n_eval)
        s_mal = -meta.score_samples(np.column_stack([pa_m, pb_m]))
        s_ben = -meta.score_samples(np.column_stack([pa_be, pb_be]))
        yp = np.concatenate([np.ones(n), np.zeros(n_eval)])
        sp = np.concatenate([s_mal, s_ben])
        out[f] = float(roc_auc_score(yp, sp))
    return out


res = {'exact': {f: [] for f in CANON_R}, 'hardened_r1': {f: [] for f in CANON_R}}
for seed in SEEDS:
    for mode in ('exact', 'hardened_r1'):
        d = run_fusion(mode, seed)
        for f, v in d.items():
            res[mode][f].append(v)
    print(f'seed{seed} done', flush=True)

summary = {'description': ('Fused B_learned AUC under exact-hash vs r=1 Hamming-hardened '
                           'MLRan grouping. RanSMAP channel unaffected (no duplicates).'),
           'per_family': {}, 'macro': {}}
for mode in res:
    fam_means = {f: round(float(np.mean(v)), 4) for f, v in res[mode].items()}
    macro = round(float(np.mean([np.mean(v) for v in res[mode].values()])), 4)
    summary['per_family'][mode] = fam_means
    summary['macro'][mode] = macro
with open(OUT, 'w') as f:
    json.dump({**summary,
               'raw': {m: {f: [round(x,4) for x in v] for f, v in res[m].items()} for m in res}},
              f, indent=1)
print(json.dumps(summary, indent=1))
print('Saved ->', OUT)
