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
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import norm, spearmanr
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import CANONICAL_FAMILIES, K_MLRAN, K_RANSMAP, META_REG_COVAR, N_INIT, RESULTS_DIR, SEEDS, TEST_SIZE, selected_meta_k  # noqa: E402
from utils import MLRAN_PATH, MLRAN_EXCLUDE, load_ransmap_clean, fit_gm, minmax_norm, oof_benign_norm_scores_mlran, oof_benign_norm_scores_ransmap  # noqa: E402

OUT = RESULTS_DIR / "channel_correlation_sensitivity.json"

META_K = selected_meta_k()
RHOS = [-0.6, -0.3, 0.0, 0.3, 0.6]
MLRAN_FAMILY_MAP = {f: f.lower() for f in CANONICAL_FAMILIES}
RANSMAP_FAMILY_MAP = {f: f for f in CANONICAL_FAMILIES}


def score(gm, X):
    return -gm.score_samples(X)


def correlated_pair(a, b, rng, n, rho):
    a_sorted, b_sorted = np.sort(a), np.sort(b)
    z0 = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    za = z0
    zb = rho * z0 + np.sqrt(max(1 - rho**2, 0.0)) * z1
    ua, ub = norm.cdf(za), norm.cdf(zb)
    ia = np.clip((ua * len(a_sorted)).astype(int), 0, len(a_sorted) - 1)
    ib = np.clip((ub * len(b_sorted)).astype(int), 0, len(b_sorted) - 1)
    return a_sorted[ia], b_sorted[ib]


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

ransmap, rfc = load_ransmap_clean(drop_zero_var=True)
r_ben = ransmap[ransmap["label"] == 0].reset_index(drop=True)
r_mal = {
    f: ransmap[ransmap["family"] == a].reset_index(drop=True)
    for f, a in RANSMAP_FAMILY_MAP.items()
}

runs = {rho: {f: [] for f in CANONICAL_FAMILIES} for rho in RHOS}
realized_rho = {rho: [] for rho in RHOS}

for seed in SEEDS:
    rng_deploy, rng_meta, rng_split, rng_pair = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    )

    tr_gids, _ = train_test_split(m_unique_gids, test_size=TEST_SIZE, random_state=seed)
    m_tr_mask = m_ben["_gid"].isin(set(tr_gids))
    Xmb_tr = m_ben[m_tr_mask][mfc].fillna(0.0).values
    Xmb_te = m_ben[~m_tr_mask][mfc].fillna(0.0).values
    gids_tr = m_ben[m_tr_mask]["_gid"].values
    gm_m = fit_gm(Xmb_tr, n_components=K_MLRAN, covariance_type="diag", seed=seed)
    sm_tr = score(gm_m, Xmb_tr)

    # stratify + StandardScaler on RanSMAP benign train
    Xrb_tr_raw, Xrb_te_raw, fam_r_tr, _ = train_test_split(
        r_ben[rfc].fillna(0.0).values,
        r_ben["family"].values,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=r_ben["family"].values,
    )
    scaler_r = StandardScaler().fit(Xrb_tr_raw)
    Xrb_tr = scaler_r.transform(Xrb_tr_raw)
    Xrb_te = scaler_r.transform(Xrb_te_raw)
    gm_r = fit_gm(Xrb_tr, n_components=K_RANSMAP, covariance_type="full", seed=seed)
    sr_tr = score(gm_r, Xrb_tr)

    sm_ben_norm_full = minmax_norm(sm_tr, score(gm_m, Xmb_te))
    sr_ben_norm_full = minmax_norm(sr_tr, score(gm_r, Xrb_te))

    # 5-fold OOF training calibration (n=600) + full 150 test evaluation
    sm_oof = oof_benign_norm_scores_mlran(Xmb_tr, gids_tr, K_MLRAN, seed)
    sr_oof = oof_benign_norm_scores_ransmap(
        Xrb_tr_raw, fam_r_tr, K_RANSMAP, seed, scale=True
    )
    n_calib = min(len(sm_oof), len(sr_oof))
    n_eval = min(len(sm_ben_norm_full), len(sr_ben_norm_full))

    for rho in RHOS:
        pa_meta, pb_meta = correlated_pair(sm_oof, sr_oof, rng_meta, n_calib, rho)
        realized_rho[rho].append(float(spearmanr(pa_meta, pb_meta).correlation))
        meta = GaussianMixture(
            META_K, covariance_type="full", reg_covar=META_REG_COVAR, n_init=N_INIT, random_state=seed
        ).fit(np.column_stack([pa_meta, pb_meta]))

        for family in CANONICAL_FAMILIES:
            Xmmal = m_mal[family][mfc].fillna(0.0).values
            Xrmal = scaler_r.transform(r_mal[family][rfc].fillna(0.0).values)
            sm_mal_norm = minmax_norm(sm_tr, score(gm_m, Xmmal))
            sr_mal_norm = minmax_norm(sr_tr, score(gm_r, Xrmal))
            n_mal = min(len(sm_mal_norm), len(sr_mal_norm))
            pa_m, pb_m = correlated_pair(sm_mal_norm, sr_mal_norm, rng_deploy, n_mal, rho)
            pa_be, pb_be = correlated_pair(
                sm_ben_norm_full, sr_ben_norm_full, rng_deploy, n_eval, rho
            )
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
        "macro_auc": macro,
        "per_family_mean_auc": per_fam,
        "realized_spearman_rho_mean": float(np.mean(realized_rho[rho])),
    }

out = {
    "description": __doc__.split("\n")[0],
    "n_seeds": len(SEEDS),
    "rhos_tested": RHOS,
    "summary": summary,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(summary, indent=2))
print("saved ->", OUT)
