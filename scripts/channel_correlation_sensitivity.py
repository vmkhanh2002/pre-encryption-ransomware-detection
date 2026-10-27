"""Fused-AUC sensitivity to the bootstrap-pairing independence assumption.

Every fused number assumes the two channels' scores can be bootstrap-paired
as if independent (paper, Section 3.5), because MLRan and RanSMAP were never
recorded on the same host. This script induces a target rank correlation rho
between the paired scores with a Gaussian-copula (NORTA) resampling scheme,
refits the same meta-GMM fusion at each rho, and reports macro AUC as a
function of rho. rho = 0 reproduces the independence result as a consistency
check; the realized Spearman correlation of each resampled pairing is
reported next to its target.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import MLRAN_CSV, RANSMAP_CSV, RESULTS_DIR, SEEDS, CANONICAL_FAMILIES, K_MLRAN, K_RANSMAP, N_INIT, META_REG_COVAR, selected_meta_k  # noqa: E402
import os, json, hashlib
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import norm, spearmanr
from utils import fit_gm, minmax_norm

MLRAN_PATH = MLRAN_CSV
RANSMAP_PATH = RANSMAP_CSV
OUT = RESULTS_DIR / 'channel_correlation_sensitivity.json'

META_K = selected_meta_k()
RHOS = [-0.6, -0.3, 0.0, 0.3, 0.6]
MLRAN_EXCLUDE = {"label", "family", "sample_id", "sha256", "split", "STRING:labels", "dataset"}
RANSMAP_EXCLUDE = {"label", "family", "trace_path", "dataset"}
MLRAN_FAMILY_MAP = {"Conti": "conti", "Ryuk": "ryuk", "LockBit": "lockbit",
                     "REvil": "revil", "WannaCry": "wannacry"}
RANSMAP_FAMILY_MAP = {f: f for f in CANONICAL_FAMILIES}


def score(gm, X):
    return -gm.score_samples(X)


def correlated_pair(a, b, rng, n, rho):
    """NORTA-style pairing: each of ia/ib is individually a valid uniform
    resample index into a's/b's own SORTED values (so returned values are
    a faithful resample of each channel's own marginal, exactly like the
    independent bootstrap pairing used elsewhere in this repo when
    rho=0), but jointly correlated via a shared Gaussian copula at target
    rho. Critically, indexing must happen into the SORTED arrays: a
    uniform percentile only corresponds to a value at that percentile
    once the array is ordered: indexing an unsorted array by a random
    position carries no relationship to the copula's rank at all."""
    a_sorted, b_sorted = np.sort(a), np.sort(b)
    z0 = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    za = z0
    zb = rho * z0 + np.sqrt(max(1 - rho ** 2, 0.0)) * z1
    ua, ub = norm.cdf(za), norm.cdf(zb)
    ia = np.clip((ua * len(a_sorted)).astype(int), 0, len(a_sorted) - 1)
    ib = np.clip((ub * len(b_sorted)).astype(int), 0, len(b_sorted) - 1)
    return a_sorted[ia], b_sorted[ib]


def split_half(arr, rng):
    idx = rng.permutation(len(arr))
    h = len(arr) // 2
    return arr[idx[:h]], arr[idx[h:]]


mlran = pd.read_csv(MLRAN_PATH, index_col=0)
mfc = [c for c in mlran.columns if c not in MLRAN_EXCLUDE]
m_ben_raw = mlran[mlran['label'] == 0].reset_index(drop=True)
m_gids = np.array([hashlib.md5(row.astype(np.float32).tobytes()).hexdigest()
                   for row in m_ben_raw[mfc].fillna(0).values])
m_ben = m_ben_raw.copy(); m_ben['_gid'] = m_gids
m_unique_gids = np.unique(m_gids)
m_mal = {f: mlran[mlran['family'] == a].reset_index(drop=True) for f, a in MLRAN_FAMILY_MAP.items()}

ransmap = pd.read_csv(RANSMAP_PATH)
num_cols = set(ransmap.select_dtypes(include='number').columns)
rfc = [c for c in ransmap.columns if c in num_cols and c not in RANSMAP_EXCLUDE]
r_ben = ransmap[ransmap['label'] == 0].reset_index(drop=True)
r_mal = {f: ransmap[ransmap['family'] == a].reset_index(drop=True) for f, a in RANSMAP_FAMILY_MAP.items()}

runs = {rho: {f: [] for f in CANONICAL_FAMILIES} for rho in RHOS}
realized_rho = {rho: [] for rho in RHOS}

for seed in SEEDS:
    rng_deploy, rng_meta, rng_split, rng_pair = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4))

    tr_gids, te_gids = train_test_split(m_unique_gids, test_size=0.2, random_state=seed)
    m_tr_mask = m_ben['_gid'].isin(set(tr_gids))
    Xmb_tr = m_ben[m_tr_mask][mfc].fillna(0.0).values
    Xmb_te = m_ben[~m_tr_mask][mfc].fillna(0.0).values
    gm_m = fit_gm(Xmb_tr, n_components=K_MLRAN, covariance_type='diag', seed=seed)
    sm_tr = score(gm_m, Xmb_tr)

    Xrb_tr, Xrb_te = train_test_split(r_ben[rfc].fillna(0.0).values, test_size=0.2,
                                       random_state=seed, stratify=r_ben['family'].values)
    gm_r = fit_gm(Xrb_tr, n_components=K_RANSMAP, covariance_type='full', seed=seed)
    sr_tr = score(gm_r, Xrb_tr)

    sm_ben_norm_full = minmax_norm(sm_tr, score(gm_m, Xmb_te))
    sr_ben_norm_full = minmax_norm(sr_tr, score(gm_r, Xrb_te))

    for rho in RHOS:
        sm_ben_calib, sm_ben_eval = split_half(sm_ben_norm_full, rng_split)
        sr_ben_calib, sr_ben_eval = split_half(sr_ben_norm_full, rng_split)
        n_calib = min(len(sm_ben_calib), len(sr_ben_calib))
        pa_meta, pb_meta = correlated_pair(sm_ben_calib, sr_ben_calib, rng_meta, max(n_calib, 200), rho)
        realized_rho[rho].append(float(spearmanr(pa_meta, pb_meta).correlation))
        meta = GaussianMixture(META_K, covariance_type='full', reg_covar=META_REG_COVAR,
                               n_init=N_INIT, random_state=seed).fit(np.column_stack([pa_meta, pb_meta]))

        n_eval = min(len(sm_ben_eval), len(sr_ben_eval))
        for family in CANONICAL_FAMILIES:
            Xmmal = m_mal[family][mfc].fillna(0.0).values
            Xrmal = r_mal[family][rfc].fillna(0.0).values
            sm_mal_norm = minmax_norm(sm_tr, score(gm_m, Xmmal))
            sr_mal_norm = minmax_norm(sr_tr, score(gm_r, Xrmal))
            n_mal = min(len(sm_mal_norm), len(sr_mal_norm))
            pa_m, pb_m = correlated_pair(sm_mal_norm, sr_mal_norm, rng_deploy, n_mal, rho)
            pa_be, pb_be = correlated_pair(sm_ben_eval, sr_ben_eval, rng_deploy, n_eval, rho)
            s_ben = -meta.score_samples(np.column_stack([pa_be, pb_be]))
            s_mal = -meta.score_samples(np.column_stack([pa_m, pb_m]))
            y = np.array([0] * n_eval + [1] * n_mal)
            s = np.concatenate([s_ben, s_mal])
            runs[rho][family].append(float(roc_auc_score(y, s)))
    print(f"seed {seed} done", flush=True)

summary = {}
for rho in RHOS:
    per_fam = {f: float(np.mean(runs[rho][f])) for f in CANONICAL_FAMILIES}
    macro = float(np.mean([v for f in CANONICAL_FAMILIES for v in runs[rho][f]]))
    summary[str(rho)] = {
        'macro_auc': macro,
        'per_family_mean_auc': per_fam,
        'realized_spearman_rho_mean': float(np.mean(realized_rho[rho])),
    }

out = {
    'description': __doc__.strip().split('\n')[0],
    'n_seeds': len(SEEDS),
    'rhos_tested': RHOS,
    'summary': summary,
    'note': ("rho=0.0 is the independence baseline (should closely match the "
             "paper's existing bootstrap-pairing result); realized_spearman_rho_mean "
             "is the EMPIRICAL correlation actually achieved by the copula "
             "resampling, reported directly rather than assumed equal to the "
             "target rho."),
}
with open(OUT, 'w') as f:
    json.dump(out, f, indent=1)
print(json.dumps(summary, indent=1))
print('saved ->', OUT)
