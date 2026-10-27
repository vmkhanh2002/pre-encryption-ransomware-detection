"""Fused system under a white-box attack on the RanSMAP channel only.

MLRan GMM (K_MLRAN, diagonal) left clean; RanSMAP GMM (K_RANSMAP, full,
standardized features) under the Level-3 attack that moves each malware
sample toward its nearest benign component in Mahalanobis distance. Scores
are fused with the learned meta-GMM calibrated on 5-fold out-of-fold benign
training scores (the `meta_K*` variant of learned_fusion.py); the whole
benign test pool is used for evaluation. Reports AUC and Deploy-95 TPR/FPR,
clean and attacked.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import CANONICAL_FAMILIES, K_MLRAN, K_RANSMAP, META_REG_COVAR, N_INIT, REG_COVAR, RESULTS_DIR, SEEDS, TEST_SIZE, selected_meta_k  # noqa: E402
from utils import MLRAN_PATH, MLRAN_EXCLUDE, load_ransmap_clean, fit_gm, minmax_norm, mahal_sq_to_centers, oof_benign_norm_scores_mlran, oof_benign_norm_scores_ransmap  # noqa: E402

OUT = RESULTS_DIR / "fused_under_attack_learned.json"
CANONICAL = CANONICAL_FAMILIES
MLRAN_FAMILY_MAP = {f: f.lower() for f in CANONICAL_FAMILIES}
META_K = selected_meta_k()

df_r, feat_cols = load_ransmap_clean(drop_zero_var=True)
Xr_raw = df_r[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
fam_r = df_r["family"].to_numpy()
is_benign_r = df_r["label"].to_numpy() == 0

mlran = pd.read_csv(MLRAN_PATH, index_col=0)
mfc = [c for c in mlran.columns if c not in MLRAN_EXCLUDE]
m_ben_raw = mlran[mlran["label"] == 0].reset_index(drop=True)
m_gids = np.array([
    hashlib.md5(row.astype(np.float32).tobytes()).hexdigest()
    for row in m_ben_raw[mfc].fillna(0).values
])
m_ben = m_ben_raw.copy()
m_ben["_gid"] = m_gids
m_unique_gids = np.unique(m_gids)
m_mal = {
    f: mlran[mlran["family"] == a].reset_index(drop=True)
    for f, a in MLRAN_FAMILY_MAP.items()
}


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
    lo = np.zeros(n)
    hi = np.full(n, max_a)
    for _ in range(30):
        mid = (lo + hi) / 2
        ok = f(mid) <= tau
        hi = np.where(ok & todo, mid, hi)
        lo = np.where(~ok & todo, mid, lo)
    alphas = np.zeros(n)
    alphas[todo] = hi[todo]
    return alphas


def pair_no_inflate(a, b, rng, n):
    ia = rng.choice(len(a), n, replace=(len(a) < n))
    ib = rng.choice(len(b), n, replace=(len(b) < n))
    return a[ia], b[ib]


res = {"description": __doc__.split("\n")[0], "per_seed": {}}
agg_clean, agg_attacked = [], []
agg_clean_deploy95, agg_attacked_deploy95 = [], []
agg_fpr95 = []

for seed in SEEDS:
    rng_pair, rng_split, rng_meta = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )

    # MLRan channel
    tr_gids, _ = train_test_split(m_unique_gids, test_size=TEST_SIZE, random_state=seed)
    m_tr_mask = m_ben["_gid"].isin(set(tr_gids))
    Xmb_tr = m_ben[m_tr_mask][mfc].fillna(0.0).values
    gids_tr = m_ben[m_tr_mask]["_gid"].values
    gm_m = fit_gm(Xmb_tr, n_components=K_MLRAN, covariance_type="diag", seed=seed)
    sm_tr = nll(gm_m, Xmb_tr)

    # RanSMAP channel (stratify + StandardScaler on benign train)
    idx_b = np.where(is_benign_r)[0]
    tr_b, te_b = train_test_split(
        idx_b, test_size=TEST_SIZE, random_state=seed, stratify=fam_r[idx_b]
    )
    scaler_r = StandardScaler().fit(Xr_raw[tr_b])
    Xr = scaler_r.transform(Xr_raw)
    gm_r = GaussianMixture(
        K_RANSMAP, covariance_type="full", reg_covar=REG_COVAR, n_init=N_INIT, random_state=seed
    ).fit(Xr[tr_b])
    sr_tr = nll(gm_r, Xr[tr_b])
    tau_r = float(np.median(sr_tr))
    mus = gm_r.means_

    mal_idx = np.array([i for i in np.where(~is_benign_r)[0] if fam_r[i] in CANONICAL])
    Xrmal = Xr[mal_idx]
    fam_mal = fam_r[mal_idx]

    # Mahalanobis distance using gm_r.precisions_
    kstar = mahal_sq_to_centers(Xrmal, gm_r).argmin(axis=1)
    T_mode = mus[kstar]
    alpha = solve_alpha(gm_r, Xrmal, T_mode, tau_r)
    Xrmal_attacked = Xrmal + alpha[:, None] * (T_mode - Xrmal)

    sr_mal_clean_n = minmax_norm(sr_tr, nll(gm_r, Xrmal))
    sr_mal_attacked_n = minmax_norm(sr_tr, nll(gm_r, Xrmal_attacked))
    sr_ben_te_n_full = minmax_norm(sr_tr, nll(gm_r, Xr[te_b]))
    sm_ben_n_full = minmax_norm(
        sm_tr, nll(gm_m, m_ben[~m_tr_mask][mfc].fillna(0.0).values)
    )

    # 5-fold OOF training calibration (n=600, no replacement) + full 150 test eval
    sm_oof = oof_benign_norm_scores_mlran(Xmb_tr, gids_tr, K_MLRAN, seed)
    sr_oof = oof_benign_norm_scores_ransmap(
        Xr_raw[tr_b], fam_r[tr_b], K_RANSMAP, seed, scale=True
    )
    n_calib_p = min(len(sm_oof), len(sr_oof))
    pa_c, pb_c = pair_no_inflate(sm_oof, sr_oof, rng_meta, n_calib_p)
    meta = GaussianMixture(
        META_K, covariance_type="full", reg_covar=META_REG_COVAR, n_init=N_INIT, random_state=seed
    ).fit(np.column_stack([pa_c, pb_c]))

    n_eval_p = min(len(sm_ben_n_full), len(sr_ben_te_n_full))
    s_ben_calib_for_thr = -meta.score_samples(np.column_stack([pa_c, pb_c]))
    thr95 = float(np.percentile(s_ben_calib_for_thr, 95))

    rng = np.random.default_rng(seed)
    seed_out = {}
    for family in CANONICAL:
        Xmmal = m_mal[family][mfc].fillna(0.0).values
        sm_mal_n = minmax_norm(sm_tr, nll(gm_m, Xmmal))
        idxf = np.where(fam_mal == family)[0]
        n_pair = min(len(sm_mal_n), len(idxf))
        pa, pb_clean = pair_no_inflate(sm_mal_n, sr_mal_clean_n[idxf], rng, n_pair)
        _, pb_atk = pair_no_inflate(sm_mal_n, sr_mal_attacked_n[idxf], rng, n_pair)

        pab, pbb = pair_no_inflate(sm_ben_n_full, sr_ben_te_n_full, rng, n_eval_p)

        s_ben_meta = -meta.score_samples(np.column_stack([pab, pbb]))
        s_mal_clean_meta = -meta.score_samples(np.column_stack([pa, pb_clean]))
        s_mal_atk_meta = -meta.score_samples(np.column_stack([pa, pb_atk]))

        y = np.concatenate([np.zeros(n_eval_p), np.ones(n_pair)])
        s_clean = np.concatenate([s_ben_meta, s_mal_clean_meta])
        s_atk = np.concatenate([s_ben_meta, s_mal_atk_meta])
        auc_clean = roc_auc_score(y, s_clean)
        auc_atk = roc_auc_score(y, s_atk)

        tpr95_clean = float(np.mean(s_mal_clean_meta >= thr95))
        tpr95_atk = float(np.mean(s_mal_atk_meta >= thr95))
        fpr95 = float(np.mean(s_ben_meta >= thr95))

        seed_out[family] = {
            "auc_clean": float(auc_clean),
            "auc_attacked": float(auc_atk),
            "deploy95_tpr_clean": tpr95_clean,
            "deploy95_tpr_attacked": tpr95_atk,
            "deploy95_fpr_clean": fpr95,
        }
        agg_clean.append(auc_clean)
        agg_attacked.append(auc_atk)
        agg_clean_deploy95.append(tpr95_clean)
        agg_attacked_deploy95.append(tpr95_atk)
        agg_fpr95.append(fpr95)

    res["per_seed"][f"seed{seed}"] = seed_out
    print(
        f"seed {seed}: "
        + " | ".join(
            f"{fm}: clean={seed_out[fm]['auc_clean']:.3f} atk={seed_out[fm]['auc_attacked']:.3f}"
            for fm in CANONICAL
        ),
        flush=True,
    )

res["summary"] = {
    "b_learned_auc_clean_mean": float(np.mean(agg_clean)),
    "b_learned_auc_attacked_mean": float(np.mean(agg_attacked)),
    "b_learned_deploy95_tpr_clean_mean": float(np.mean(agg_clean_deploy95)),
    "b_learned_deploy95_tpr_attacked_mean": float(np.mean(agg_attacked_deploy95)),
    "b_learned_deploy95_fpr_clean_mean": float(np.mean(agg_fpr95)),
}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res["summary"], indent=2))
print("saved ->", OUT)
