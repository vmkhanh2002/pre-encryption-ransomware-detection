"""Learned score fusion: a density model over the 2-D benign score cloud.

Instead of a fixed algebraic rule (mean, max, min, product), the benign
MinMax-normalized score pairs [s_mlran, s_ransmap] are bootstrap-paired
(the channels are not sample-aligned), a small GMM is fitted to that benign
cloud, and any sample is scored by its negative log-likelihood under it.
This stays label-free, captures cross-channel correlation, and extends to
more than two channels.

- No train/eval leakage: each channel's benign test pool is split into a
  disjoint calibration half (fits the meta-GMM; its component count is
  chosen there by BIC over META_K_CANDIDATES) and an evaluation half (paired
  with malware, never seen by the meta-GMM).
- The fixed product rule is evaluated on the same 5-family / 10-seed harness
  as a reference; it needs no calibration, so it uses the full benign test
  pool.

Writes learned_fusion.json, which later scripts read for the selected
meta-GMM component count (paper, Table 3).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import MLRAN_CSV, RANSMAP_CSV, RESULTS_DIR, SEEDS, CANONICAL_FAMILIES, K_MLRAN, K_RANSMAP, META_K_CANDIDATES, META_REG_COVAR, N_INIT  # noqa: E402
import os, json, hashlib
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             precision_recall_fscore_support, matthews_corrcoef)
from utils import fit_gm, minmax_norm

MLRAN_PATH = MLRAN_CSV
RANSMAP_PATH = RANSMAP_CSV
OUT = RESULTS_DIR / 'learned_fusion.json'

MLRAN_EXCLUDE = {"label", "family", "sample_id", "sha256", "split", "STRING:labels", "dataset"}
RANSMAP_EXCLUDE = {"label", "family", "trace_path", "dataset"}
MLRAN_FAMILY_MAP = {"Conti": "conti", "Ryuk": "ryuk", "LockBit": "lockbit",
                     "REvil": "revil", "WannaCry": "wannacry"}
RANSMAP_FAMILY_MAP = {"Conti": "Conti", "Ryuk": "Ryuk", "LockBit": "LockBit",
                       "REvil": "REvil", "WannaCry": "WannaCry"}
META_KS = META_K_CANDIDATES

def score(gm, X):
    return -gm.score_samples(X)

def pair_scores(a, b, rng, n):
    ia = rng.choice(len(a), n, replace=(len(a) < n))
    ib = rng.choice(len(b), n, replace=(len(b) < n))
    return a[ia], b[ib]

def compute_metrics(y, s, s_ben_eval=None, s_mal=None, thr95=None):
    fpr, tpr, thr = roc_curve(y, s)
    auc = float(roc_auc_score(y, s))
    auprc = float(average_precision_score(y, s))
    j = int(np.argmax(tpr - fpr))
    y_hat = (s >= thr[j]).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, y_hat, average='binary', zero_division=0)
    mcc = float(matthews_corrcoef(y, y_hat))
    out = {'auc': auc, 'auprc': auprc, 'f1_opt': float(f1),
           'precision_opt': float(pr), 'recall_opt': float(rc), 'mcc_opt': mcc,
           'tpr_at_fpr1': float(np.interp(0.01, fpr, tpr)),
           'tpr_at_fpr5': float(np.interp(0.05, fpr, tpr))}
    if thr95 is not None:
        out['deploy95_tpr'] = float(np.mean(s_mal >= thr95))
        out['deploy95_fpr'] = float(np.mean(s_ben_eval >= thr95))
    return out

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

runs = {f'meta_K{k}': {f: [] for f in CANONICAL_FAMILIES} for k in META_KS}
runs['B_product_ref'] = {f: [] for f in CANONICAL_FAMILIES}
# RanSMAP-alone per-seed-per-family AUC, kept OUTSIDE `runs`/`macro()` (single-
# channel, no calibration, no thr95) so it never touches the existing `summary`
# dict. Purely additive, for the significance test comparing B_learned vs a
# single channel (see `learned_fusion_significance.py`).
ransmap_alone_folds = {f: [] for f in CANONICAL_FAMILIES}
# BIC of each candidate meta-GMM on its own calibration-half fit, accumulated
# across seeds. Selects K by an in-sample criterion computed only on the
# calibration half, instead of picking whichever K happened to score best on
# the eval half (the latter is a mild eval-side model-selection leak).
bic_accum = {k: [] for k in META_KS}

for seed in SEEDS:
    rng_deploy, rng_meta, rng_split, rng_thr = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4))

    tr_gids, te_gids = train_test_split(m_unique_gids, test_size=0.2, random_state=seed)
    m_tr_mask = m_ben['_gid'].isin(set(tr_gids))
    Xmb_tr = m_ben[m_tr_mask][mfc].fillna(0.0).values
    Xmb_te = m_ben[~m_tr_mask][mfc].fillna(0.0).values

    Xrb_tr, Xrb_te = train_test_split(r_ben[rfc].fillna(0.0).values, test_size=0.2,
                                       random_state=seed, stratify=r_ben['family'].values)

    gm_m = fit_gm(Xmb_tr, n_components=K_MLRAN, covariance_type='diag', seed=seed)
    gm_r = fit_gm(Xrb_tr, n_components=K_RANSMAP, covariance_type='full', seed=seed)
    sm_tr, sr_tr = score(gm_m, Xmb_tr), score(gm_r, Xrb_tr)

    sm_ben_norm_full = minmax_norm(sm_tr, score(gm_m, Xmb_te))
    sr_ben_norm_full = minmax_norm(sr_tr, score(gm_r, Xrb_te))

    # disjoint calib/eval split of the benign TEST pool (per channel, independently)
    sm_ben_calib, sm_ben_eval = split_half(sm_ben_norm_full, rng_split)
    sr_ben_calib, sr_ben_eval = split_half(sr_ben_norm_full, rng_split)

    # meta-model fit ONLY on calibration-half bootstrap pairs
    n_calib_p = min(len(sm_ben_calib), len(sr_ben_calib))
    pa_meta, pb_meta = pair_scores(sm_ben_calib, sr_ben_calib, rng_meta, max(n_calib_p, 200))
    XY_calib = np.column_stack([pa_meta, pb_meta])
    meta_models = {}
    for k in META_KS:
        meta_models[k] = GaussianMixture(k, covariance_type='full', reg_covar=META_REG_COVAR,
                                         n_init=N_INIT, random_state=seed).fit(XY_calib)
        bic_accum[k].append(float(meta_models[k].bic(XY_calib)))

    n_eval_p = min(len(sm_ben_eval), len(sr_ben_eval))

    # Deploy-95 thresholds: 95th percentile of a train-like/calibration
    # benign pool, never touching the eval pool used for TPR/FPR below
    thr95_meta = {k: float(np.percentile(-meta_models[k].score_samples(XY_calib), 95))
                  for k in META_KS}
    n_ben_p = min(len(sm_ben_norm_full), len(sr_ben_norm_full))
    sm_tr_n, sr_tr_n = minmax_norm(sm_tr, sm_tr), minmax_norm(sr_tr, sr_tr)
    pa_thr_n, pb_thr_n = pair_scores(sm_tr_n, sr_tr_n, rng_thr, n_ben_p)
    thr95_bproduct = float(np.percentile(pa_thr_n * pb_thr_n, 95))

    for family in CANONICAL_FAMILIES:
        Xmmal = m_mal[family][mfc].fillna(0.0).values
        Xrmal = r_mal[family][rfc].fillna(0.0).values
        sm_mal_norm = minmax_norm(sm_tr, score(gm_m, Xmmal))
        sr_mal_norm = minmax_norm(sr_tr, score(gm_r, Xrmal))

        # RanSMAP-alone AUC for this seed/family: single channel, no bootstrap
        # pairing needed (not cross-channel), so this draws NO random numbers
        # and cannot perturb rng_deploy/rng_meta/rng_split/rng_thr's sequence
        # (verified: only reads already-materialized sr_ben_eval/sr_mal_norm).
        y_r_alone = np.array([0] * len(sr_ben_eval) + [1] * len(sr_mal_norm))
        s_r_alone = np.concatenate([sr_ben_eval, sr_mal_norm])
        ransmap_alone_folds[family].append(float(roc_auc_score(y_r_alone, s_r_alone)))

        n_mal_p = min(len(sm_mal_norm), len(sr_mal_norm))
        pa_m, pb_m = pair_scores(sm_mal_norm, sr_mal_norm, rng_deploy, n_mal_p)
        # meta models: evaluated ONLY on the held-out eval half (never seen by the meta-model)
        pa_be, pb_be = pair_scores(sm_ben_eval, sr_ben_eval, rng_deploy, n_eval_p)
        XY_mal = np.column_stack([pa_m, pb_m])
        XY_ben_eval = np.column_stack([pa_be, pb_be])
        y_eval = np.array([0]*n_eval_p + [1]*n_mal_p)
        for k in META_KS:
            s_meta_ben = -meta_models[k].score_samples(XY_ben_eval)
            s_meta_mal = -meta_models[k].score_samples(XY_mal)
            s_meta = np.concatenate([s_meta_ben, s_meta_mal])
            runs[f'meta_K{k}'][family].append(compute_metrics(
                y_eval, s_meta, s_ben_eval=s_meta_ben, s_mal=s_meta_mal, thr95=thr95_meta[k]))

        # B_product: no calibration needed, so scored on the FULL benign test pool (fair "no cost" reference)
        n_ben_p2 = min(len(sm_ben_norm_full), len(sr_ben_norm_full))
        pa_bf, pb_bf = pair_scores(sm_ben_norm_full, sr_ben_norm_full, rng_deploy, n_ben_p2)
        s_product_ben = pa_bf * pb_bf
        s_product_mal = pa_m * pb_m
        s_product = np.concatenate([s_product_ben, s_product_mal])
        y_full = np.array([0]*n_ben_p2 + [1]*n_mal_p)
        runs['B_product_ref'][family].append(compute_metrics(
            y_full, s_product, s_ben_eval=s_product_ben, s_mal=s_product_mal, thr95=thr95_bproduct))

    print(f"seed {seed} done", flush=True)

def macro(strategy_runs):
    per_fam = {f: float(np.mean([r['auc'] for r in strategy_runs[f]])) for f in CANONICAL_FAMILIES}
    keys = ['auc', 'auprc', 'f1_opt', 'precision_opt', 'recall_opt', 'mcc_opt',
            'tpr_at_fpr1', 'tpr_at_fpr5', 'deploy95_tpr', 'deploy95_fpr']
    out = {'per_family_mean_auc': per_fam}
    for k in keys:
        vals = [r[k] for f in CANONICAL_FAMILIES for r in strategy_runs[f]]
        out[f'macro_{k}'] = float(np.mean(vals))
    return out

summary = {name: macro(sruns) for name, sruns in runs.items()}

# K selection: argmin of mean BIC on the calibration-half fit, across the 10
# seeds. An in-sample criterion computed only on data the meta-model itself
# was fit on, replacing the previous "report whichever K scored best on the
# held-out eval half" choice (that picks K using the same data used to report
# its performance, a mild eval-side model-selection leak).
mean_bic = {k: float(np.mean(v)) for k, v in bic_accum.items()}
K_STAR = min(mean_bic, key=mean_bic.get)
meta_k_selection = {
    'method': 'argmin mean BIC on calibration-half XY_calib, averaged across 10 seeds',
    'per_seed_bic': {f'meta_K{k}': v for k, v in bic_accum.items()},
    'mean_bic': {f'meta_K{k}': v for k, v in mean_bic.items()},
    'selected_k': K_STAR,
}

# Additive raw per-seed-per-family AUC arrays, for the family-blocked
# significance test in learned_fusion_significance.py. Does not touch
# `summary` above in any way.
raw_per_seed_auc = {'ransmap_alone': ransmap_alone_folds}
for k in META_KS:
    raw_per_seed_auc[f'meta_K{k}'] = {
        f: [r['auc'] for r in runs[f'meta_K{k}'][f]] for f in CANONICAL_FAMILIES
    }

out = {'description': __doc__.split('\n')[0], 'n_seeds': len(SEEDS),
       'canonical_families': CANONICAL_FAMILIES, 'meta_ks_tested': META_KS,
       'note': 'B_product_ref evaluated on the FULL benign test pool (needs no calibration); '
               'meta_K* evaluated ONLY on the held-out eval-half never seen during meta-model fitting.',
       'summary': summary,
       'meta_k_selection': meta_k_selection,
       'raw_per_seed_auc': raw_per_seed_auc}
with open(OUT, 'w') as f:
    json.dump(out, f, indent=1)
print(json.dumps(summary, indent=1))
print('BIC-selected K:', K_STAR, '(mean BIC per K:', mean_bic, ')')
print('saved ->', OUT)
