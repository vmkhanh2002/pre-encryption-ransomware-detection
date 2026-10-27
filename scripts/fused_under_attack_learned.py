"""Fused system under a white-box attack on the RanSMAP channel only.

MLRan GMM (K_MLRAN, diagonal) left clean; RanSMAP GMM (K_RANSMAP, full,
raw features) under the Level-3 nearest-benign-mode attack. Scores are fused
with the learned joint-density model, using the same leak-free split as
learned_fusion.py: each channel's benign test pool is split into a
calibration half (fits the meta-GMM, with the component count selected
there) and an evaluation half (paired with malware, never seen during
calibration). Reports AUC and Deploy-95 TPR/FPR, clean and attacked.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import MLRAN_CSV, RESULTS_DIR, SEEDS, K_MLRAN, K_RANSMAP, COV_RANSMAP, REG_COVAR, N_INIT, META_REG_COVAR, selected_meta_k  # noqa: E402
import os, json, hashlib
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import load_ransmap, fit_gm, minmax_norm

MLRAN_PATH = MLRAN_CSV
OUT = RESULTS_DIR / 'fused_under_attack_learned.json'
CANONICAL = ['Conti', 'Ryuk', 'LockBit', 'REvil', 'WannaCry']
MLRAN_EXCLUDE = {"label", "family", "sample_id", "sha256", "split", "STRING:labels", "dataset"}
MLRAN_FAMILY_MAP = {"Conti": "conti", "Ryuk": "ryuk", "LockBit": "lockbit",
                     "REvil": "revil", "WannaCry": "wannacry"}
META_K = selected_meta_k()

df_r, feat_cols = load_ransmap()
Xr = df_r[feat_cols].to_numpy(dtype=np.float64)
fam_r = df_r['family'].to_numpy()
is_benign_r = df_r['label'].to_numpy() == 0

mlran = pd.read_csv(MLRAN_PATH, index_col=0)
mfc = [c for c in mlran.columns if c not in MLRAN_EXCLUDE]
m_ben_raw = mlran[mlran['label'] == 0].reset_index(drop=True)
m_gids = np.array([hashlib.md5(row.astype(np.float32).tobytes()).hexdigest()
                   for row in m_ben_raw[mfc].fillna(0).values])
m_ben = m_ben_raw.copy(); m_ben['_gid'] = m_gids
m_unique_gids = np.unique(m_gids)
m_mal = {f: mlran[mlran['family'] == a].reset_index(drop=True)
          for f, a in MLRAN_FAMILY_MAP.items()}

def nll(gm, Xq):
    return -gm.score_samples(Xq)

def solve_alpha(gm, X0, T, tau, max_a=1.0):
    def f(a):
        aa = np.asarray(a)
        if aa.ndim == 0:
            return nll(gm, X0 + float(aa) * (T - X0))
        return nll(gm, X0 + aa[:, None] * (T - X0))
    n = len(X0)
    inside = f(0.0) <= tau
    reachable = f(max_a) <= tau
    todo = (~inside) & reachable
    lo = np.zeros(n); hi = np.full(n, max_a)
    for _ in range(30):
        mid = (lo + hi) / 2
        ok = f(mid) <= tau
        hi = np.where(ok & todo, mid, hi)
        lo = np.where(~ok & todo, mid, lo)
    alphas = np.zeros(n); alphas[todo] = hi[todo]
    return alphas

def pair(a, b, rng, n):
    ia = rng.choice(len(a), n, replace=(len(a) < n))
    ib = rng.choice(len(b), n, replace=(len(b) < n))
    return a[ia], b[ib]

def split_half(arr, rng):
    idx = rng.permutation(len(arr))
    h = len(arr) // 2
    return arr[idx[:h]], arr[idx[h:]]

res = {'description': __doc__.split('\n')[0], 'per_seed': {}}
agg_clean, agg_attacked = [], []
agg_clean_deploy95, agg_attacked_deploy95 = [], []
agg_fpr95 = []

for seed in SEEDS:
    rng_pair, rng_split, rng_meta = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))

    # MLRan channel (clean, matches ano_combined_eval.py)
    tr_gids, te_gids = train_test_split(m_unique_gids, test_size=0.2, random_state=seed)
    m_tr_mask = m_ben['_gid'].isin(set(tr_gids))
    Xmb_tr = m_ben[m_tr_mask][mfc].fillna(0.0).values
    gm_m = fit_gm(Xmb_tr, n_components=K_MLRAN, covariance_type='diag', seed=seed)
    sm_tr = nll(gm_m, Xmb_tr)

    # RanSMAP channel (raw features, as in the white-box attack scripts)
    idx_b = np.where(is_benign_r)[0]
    tr_b, te_b = train_test_split(idx_b, test_size=0.2, random_state=seed)
    gm_r = GaussianMixture(K_RANSMAP, covariance_type=COV_RANSMAP, reg_covar=REG_COVAR,
                           n_init=N_INIT, random_state=seed).fit(Xr[tr_b])
    sr_tr = nll(gm_r, Xr[tr_b])
    tau_r = float(np.median(sr_tr))
    mus, covs = gm_r.means_, gm_r.covariances_

    mal_idx = np.array([i for i in np.where(~is_benign_r)[0] if fam_r[i] in CANONICAL])
    Xrmal = Xr[mal_idx]
    fam_mal = fam_r[mal_idx]

    def mahal_to_centers(Xq):
        D = np.empty((len(Xq), len(mus)))
        for kk in range(len(mus)):
            diff = Xq - mus[kk]
            D[:, kk] = np.einsum('ij,jk,ik->i', diff, covs[kk], diff)
        return D
    kstar = mahal_to_centers(Xrmal).argmin(axis=1)
    T_mode = mus[kstar]
    alpha = solve_alpha(gm_r, Xrmal, T_mode, tau_r)
    Xrmal_attacked = Xrmal + alpha[:, None] * (T_mode - Xrmal)

    sr_mal_clean_n = minmax_norm(sr_tr, nll(gm_r, Xrmal))
    sr_mal_attacked_n = minmax_norm(sr_tr, nll(gm_r, Xrmal_attacked))
    sr_ben_te_n_full = minmax_norm(sr_tr, nll(gm_r, Xr[te_b]))
    sm_ben_n_full = minmax_norm(sm_tr, nll(gm_m, m_ben[~m_tr_mask][mfc].fillna(0.0).values))

    # leak-free calib/eval split of benign TEST scores, per channel
    sm_ben_calib, sm_ben_eval = split_half(sm_ben_n_full, rng_split)
    sr_ben_calib, sr_ben_eval = split_half(sr_ben_te_n_full, rng_split)

    n_calib_p = min(len(sm_ben_calib), len(sr_ben_calib))
    pa_c, pb_c = pair(sm_ben_calib, sr_ben_calib, rng_meta, max(n_calib_p, 200))
    meta = GaussianMixture(META_K, covariance_type='full', reg_covar=META_REG_COVAR,
                           n_init=N_INIT, random_state=seed).fit(np.column_stack([pa_c, pb_c]))

    n_eval_p = min(len(sm_ben_eval), len(sr_ben_eval))

    rng = np.random.default_rng(seed)
    seed_out = {}
    for family in CANONICAL:
        Xmmal = m_mal[family][mfc].fillna(0.0).values
        sm_mal_n = minmax_norm(sm_tr, nll(gm_m, Xmmal))
        idxf = np.where(fam_mal == family)[0]
        n_pair = min(len(sm_mal_n), len(idxf))
        pa, pb_clean = pair(sm_mal_n, sr_mal_clean_n[idxf], rng, n_pair)
        _,  pb_atk   = pair(sm_mal_n, sr_mal_attacked_n[idxf], rng, n_pair)

        pab, pbb = pair(sm_ben_eval, sr_ben_eval, rng, n_eval_p)

        s_ben_meta = -meta.score_samples(np.column_stack([pab, pbb]))
        s_mal_clean_meta = -meta.score_samples(np.column_stack([pa, pb_clean]))
        s_mal_atk_meta = -meta.score_samples(np.column_stack([pa, pb_atk]))

        y = np.concatenate([np.zeros(n_eval_p), np.ones(n_pair)])
        s_clean = np.concatenate([s_ben_meta, s_mal_clean_meta])
        s_atk = np.concatenate([s_ben_meta, s_mal_atk_meta])
        auc_clean = roc_auc_score(y, s_clean)
        auc_atk = roc_auc_score(y, s_atk)

        # Deploy-95 threshold from the CALIBRATION-fit benign pairs (train-like,
        # never touches the eval set used for TPR/FPR below). Avoids the
        # circularity of setting and checking a percentile on the same sample.
        s_ben_calib_for_thr = -meta.score_samples(np.column_stack([pa_c, pb_c]))
        thr95 = float(np.percentile(s_ben_calib_for_thr, 95))
        tpr95_clean = float(np.mean(s_mal_clean_meta >= thr95))
        tpr95_atk = float(np.mean(s_mal_atk_meta >= thr95))
        fpr95 = float(np.mean(s_ben_meta >= thr95))

        seed_out[family] = {'auc_clean': float(auc_clean), 'auc_attacked': float(auc_atk),
                            'deploy95_tpr_clean': tpr95_clean, 'deploy95_tpr_attacked': tpr95_atk,
                            'deploy95_fpr_clean': fpr95}
        agg_clean.append(auc_clean); agg_attacked.append(auc_atk)
        agg_clean_deploy95.append(tpr95_clean); agg_attacked_deploy95.append(tpr95_atk)
        agg_fpr95.append(fpr95)

    res['per_seed'][f'seed{seed}'] = seed_out
    print(f"seed {seed}: " + " | ".join(
        f"{fm}: clean={seed_out[fm]['auc_clean']:.3f} atk={seed_out[fm]['auc_attacked']:.3f}"
        for fm in CANONICAL), flush=True)

res['summary'] = {
    'b_learned_auc_clean_mean': float(np.mean(agg_clean)),
    'b_learned_auc_attacked_mean': float(np.mean(agg_attacked)),
    'b_learned_deploy95_tpr_clean_mean': float(np.mean(agg_clean_deploy95)),
    'b_learned_deploy95_tpr_attacked_mean': float(np.mean(agg_attacked_deploy95)),
    'b_learned_deploy95_fpr_clean_mean': float(np.mean(agg_fpr95)),
}
with open(OUT, 'w') as f:
    json.dump(res, f, indent=1)
print(json.dumps(res['summary'], indent=1))
print('saved ->', OUT)
