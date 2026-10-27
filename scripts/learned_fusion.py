"""Learned score fusion: a density model over the 2-D benign score cloud.

Instead of a fixed algebraic rule (mean, max, min, product), the benign
MinMax-normalized score pairs [s_mlran, s_ransmap] are bootstrap-paired
(the channels are not sample-aligned), a small GMM is fitted to that benign
cloud, and any sample is scored by its negative log-likelihood under it.
RanSMAP features are standardized on the benign training split, and the
features that are constant over all benign traces are dropped.

The meta-GMM is calibrated in two ways, both without resampling a point
more than once:
- `meta_K*_split75_noreplace` (paper, Table 3): each channel's benign test
  pool is split into a calibration half (75 RanSMAP traces) and an evaluation
  half that the meta-GMM never sees.
- `meta_K*`: calibration on 5-fold out-of-fold benign training scores
  (n = 600), so the whole benign test pool (150) is used for evaluation.
  The adversarial experiments use this variant.
The component count is chosen by BIC over META_K_CANDIDATES on the
calibration data of each variant. The fixed product rule is evaluated on the
same harness as a reference.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import CANONICAL_FAMILIES, K_MLRAN, K_RANSMAP, META_K_CANDIDATES, META_REG_COVAR, N_INIT, RESULTS_DIR, SEEDS, TEST_SIZE  # noqa: E402
from utils import MLRAN_PATH, MLRAN_EXCLUDE, load_ransmap_clean, fit_gm, minmax_norm, oof_benign_norm_scores_mlran, oof_benign_norm_scores_ransmap  # noqa: E402

OUT = RESULTS_DIR / "learned_fusion.json"

MLRAN_FAMILY_MAP = {f: f.lower() for f in CANONICAL_FAMILIES}
RANSMAP_FAMILY_MAP = {f: f for f in CANONICAL_FAMILIES}
META_KS = META_K_CANDIDATES


def score(gm, X):
    return -gm.score_samples(X)


def pair_scores_no_inflate(a, b, rng, n):
    """Sample without replacement whenever len >= n."""
    ia = rng.choice(len(a), n, replace=(len(a) < n))
    ib = rng.choice(len(b), n, replace=(len(b) < n))
    return a[ia], b[ib]


def compute_metrics(y, s, s_ben_eval=None, s_mal=None, thr95=None):
    fpr, tpr, thr = roc_curve(y, s)
    auc = float(roc_auc_score(y, s))
    auprc = float(average_precision_score(y, s))
    j = int(np.argmax(tpr - fpr))
    y_hat = (s >= thr[j]).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, y_hat, average="binary", zero_division=0
    )
    mcc = float(matthews_corrcoef(y, y_hat))
    out = {
        "auc": auc,
        "auprc": auprc,
        "f1_opt": float(f1),
        "precision_opt": float(pr),
        "recall_opt": float(rc),
        "mcc_opt": mcc,
        "tpr_at_fpr1": float(np.interp(0.01, fpr, tpr)),
        "tpr_at_fpr5": float(np.interp(0.05, fpr, tpr)),
    }
    if thr95 is not None:
        out["deploy95_tpr"] = float(np.mean(s_mal >= thr95))
        out["deploy95_fpr"] = float(np.mean(s_ben_eval >= thr95))
    return out


def split_half(arr, rng):
    idx = rng.permutation(len(arr))
    h = len(arr) // 2
    return arr[idx[:h]], arr[idx[h:]]


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

runs = {f"meta_K{k}": {f: [] for f in CANONICAL_FAMILIES} for k in META_KS}
runs.update({f"meta_K{k}_split75_noreplace": {f: [] for f in CANONICAL_FAMILIES} for k in META_KS})
runs["B_product_ref"] = {f: [] for f in CANONICAL_FAMILIES}
ransmap_alone_folds = {f: [] for f in CANONICAL_FAMILIES}

bic_accum_oof600 = {k: [] for k in META_KS}
bic_accum_split75 = {k: [] for k in META_KS}

for seed in SEEDS:
    rng_deploy, rng_meta, rng_split, rng_thr = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    )

    tr_gids, _ = train_test_split(m_unique_gids, test_size=TEST_SIZE, random_state=seed)
    m_tr_mask = m_ben["_gid"].isin(set(tr_gids))
    Xmb_tr = m_ben[m_tr_mask][mfc].fillna(0.0).values
    Xmb_te = m_ben[~m_tr_mask][mfc].fillna(0.0).values
    gids_tr = m_ben[m_tr_mask]["_gid"].values

    # Stratified split + StandardScaler fit on benign training split only
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

    gm_m = fit_gm(Xmb_tr, n_components=K_MLRAN, covariance_type="diag", seed=seed)
    gm_r = fit_gm(Xrb_tr, n_components=K_RANSMAP, covariance_type="full", seed=seed)
    sm_tr, sr_tr = score(gm_m, Xmb_tr), score(gm_r, Xrb_tr)

    sm_ben_norm_full = minmax_norm(sm_tr, score(gm_m, Xmb_te))
    sr_ben_norm_full = minmax_norm(sr_tr, score(gm_r, Xrb_te))

    # --- Calibration on out-of-fold training scores: 5-fold OOF benign calibration on training split (n=600, no replacement) ---
    sm_oof_calib = oof_benign_norm_scores_mlran(Xmb_tr, gids_tr, K_MLRAN, seed)
    sr_oof_calib = oof_benign_norm_scores_ransmap(
        Xrb_tr_raw, fam_r_tr, K_RANSMAP, seed, scale=True
    )
    n_oof_p = min(len(sm_oof_calib), len(sr_oof_calib))  # 600 genuine out-of-fold points
    pa_oof, pb_oof = pair_scores_no_inflate(sm_oof_calib, sr_oof_calib, rng_meta, n_oof_p)
    XY_oof_calib = np.column_stack([pa_oof, pb_oof])

    meta_models_oof = {}
    for k in META_KS:
        meta_models_oof[k] = GaussianMixture(
            k, covariance_type="full", reg_covar=META_REG_COVAR, n_init=N_INIT, random_state=seed
        ).fit(XY_oof_calib)
        bic_accum_oof600[k].append(float(meta_models_oof[k].bic(XY_oof_calib)))

    thr95_meta_oof = {
        k: float(np.percentile(-meta_models_oof[k].score_samples(XY_oof_calib), 95))
        for k in META_KS
    }

    # --- Calibration on half of the benign test pool: 75/75 test split, paired without replacement ---
    sm_ben_calib, sm_ben_eval = split_half(sm_ben_norm_full, rng_split)
    sr_ben_calib, sr_ben_eval = split_half(sr_ben_norm_full, rng_split)
    n_calib_p = min(len(sm_ben_calib), len(sr_ben_calib))  # 75 points, no replacement!
    pa_split75, pb_split75 = pair_scores_no_inflate(
        sm_ben_calib, sr_ben_calib, rng_meta, n_calib_p
    )
    XY_split75_calib = np.column_stack([pa_split75, pb_split75])

    meta_models_split75 = {}
    for k in META_KS:
        meta_models_split75[k] = GaussianMixture(
            k, covariance_type="full", reg_covar=META_REG_COVAR, n_init=N_INIT, random_state=seed
        ).fit(XY_split75_calib)
        bic_accum_split75[k].append(float(meta_models_split75[k].bic(XY_split75_calib)))

    thr95_meta_split75 = {
        k: float(np.percentile(-meta_models_split75[k].score_samples(XY_split75_calib), 95))
        for k in META_KS
    }

    n_eval_split75 = min(len(sm_ben_eval), len(sr_ben_eval))
    n_ben_full = min(len(sm_ben_norm_full), len(sr_ben_norm_full))  # 150 full test samples

    pa_thr_n, pb_thr_n = pair_scores_no_inflate(sm_oof_calib, sr_oof_calib, rng_thr, n_oof_p)
    thr95_bproduct = float(np.percentile(pa_thr_n * pb_thr_n, 95))

    for family in CANONICAL_FAMILIES:
        Xmmal = m_mal[family][mfc].fillna(0.0).values
        Xrmal = scaler_r.transform(r_mal[family][rfc].fillna(0.0).values)
        sm_mal_norm = minmax_norm(sm_tr, score(gm_m, Xmmal))
        sr_mal_norm = minmax_norm(sr_tr, score(gm_r, Xrmal))

        # RanSMAP-alone AUC on the full benign test pool (n=150)
        y_r_alone = np.array([0] * len(sr_ben_norm_full) + [1] * len(sr_mal_norm))
        s_r_alone = np.concatenate([sr_ben_norm_full, sr_mal_norm])
        ransmap_alone_folds[family].append(float(roc_auc_score(y_r_alone, s_r_alone)))

        n_mal_p = min(len(sm_mal_norm), len(sr_mal_norm))
        pa_m, pb_m = pair_scores_no_inflate(sm_mal_norm, sr_mal_norm, rng_deploy, n_mal_p)
        XY_mal = np.column_stack([pa_m, pb_m])

        # Evaluate OOF-calibrated meta-models on the FULL benign test pool (n_ben_full = 150)
        pa_bf, pb_bf = pair_scores_no_inflate(
            sm_ben_norm_full, sr_ben_norm_full, rng_deploy, n_ben_full
        )
        XY_ben_full = np.column_stack([pa_bf, pb_bf])
        y_full = np.array([0] * n_ben_full + [1] * n_mal_p)

        for k in META_KS:
            s_meta_ben = -meta_models_oof[k].score_samples(XY_ben_full)
            s_meta_mal = -meta_models_oof[k].score_samples(XY_mal)
            s_meta = np.concatenate([s_meta_ben, s_meta_mal])
            runs[f"meta_K{k}"][family].append(
                compute_metrics(
                    y_full,
                    s_meta,
                    s_ben_eval=s_meta_ben,
                    s_mal=s_meta_mal,
                    thr95=thr95_meta_oof[k],
                )
            )

        # Evaluate split75-no-replacement meta-models on the 75-sample eval half
        pa_be, pb_be = pair_scores_no_inflate(
            sm_ben_eval, sr_ben_eval, rng_deploy, n_eval_split75
        )
        XY_ben_eval75 = np.column_stack([pa_be, pb_be])
        y_eval75 = np.array([0] * n_eval_split75 + [1] * n_mal_p)
        for k in META_KS:
            s_meta_ben75 = -meta_models_split75[k].score_samples(XY_ben_eval75)
            s_meta_mal75 = -meta_models_split75[k].score_samples(XY_mal)
            s_meta75 = np.concatenate([s_meta_ben75, s_meta_mal75])
            runs[f"meta_K{k}_split75_noreplace"][family].append(
                compute_metrics(
                    y_eval75,
                    s_meta75,
                    s_ben_eval=s_meta_ben75,
                    s_mal=s_meta_mal75,
                    thr95=thr95_meta_split75[k],
                )
            )

        # B_product reference on the FULL benign test pool (n=150)
        s_product_ben = pa_bf * pb_bf
        s_product_mal = pa_m * pb_m
        s_product = np.concatenate([s_product_ben, s_product_mal])
        runs["B_product_ref"][family].append(
            compute_metrics(
                y_full,
                s_product,
                s_ben_eval=s_product_ben,
                s_mal=s_product_mal,
                thr95=thr95_bproduct,
            )
        )

    print(f"seed {seed} done", flush=True)


def macro(strategy_runs):
    per_fam = {
        f: float(np.mean([r["auc"] for r in strategy_runs[f]]))
        for f in CANONICAL_FAMILIES
    }
    keys = [
        "auc",
        "auprc",
        "f1_opt",
        "precision_opt",
        "recall_opt",
        "mcc_opt",
        "tpr_at_fpr1",
        "tpr_at_fpr5",
        "deploy95_tpr",
        "deploy95_fpr",
    ]
    out = {"per_family_mean_auc": per_fam}
    for k in keys:
        vals = [r[k] for f in CANONICAL_FAMILIES for r in strategy_runs[f]]
        out[f"macro_{k}"] = float(np.mean(vals))
    return out


summary = {name: macro(sruns) for name, sruns in runs.items()}
summary["ransmap_alone_macro_auc"] = float(
    np.mean([v for f in CANONICAL_FAMILIES for v in ransmap_alone_folds[f]])
)

mean_bic_oof = {k: float(np.mean(v)) for k, v in bic_accum_oof600.items()}
mean_bic_split75 = {k: float(np.mean(v)) for k, v in bic_accum_split75.items()}
K_STAR = min(mean_bic_oof, key=mean_bic_oof.get)
K_STAR_SPLIT75 = min(mean_bic_split75, key=mean_bic_split75.get)

meta_k_selection = {
    "method": "argmin mean BIC on 5-fold OOF training calibration (n=600, no replacement)",
    "mean_bic_oof600": {f"meta_K{k}": v for k, v in mean_bic_oof.items()},
    "selected_k": K_STAR,
    "mean_bic_split75_noreplace": {f"meta_K{k}": v for k, v in mean_bic_split75.items()},
    "selected_k_split75_noreplace": K_STAR_SPLIT75,
}

raw_per_seed_auc = {"ransmap_alone": ransmap_alone_folds}
for k in META_KS:
    raw_per_seed_auc[f"meta_K{k}"] = {
        f: [r["auc"] for r in runs[f"meta_K{k}"][f]] for f in CANONICAL_FAMILIES
    }
    raw_per_seed_auc[f"meta_K{k}_split75_noreplace"] = {
        f: [r["auc"] for r in runs[f"meta_K{k}_split75_noreplace"][f]]
        for f in CANONICAL_FAMILIES
    }

out = {
    "description": __doc__.split("\n")[0],
    "n_seeds": len(SEEDS),
    "canonical_families": CANONICAL_FAMILIES,
    "meta_ks_tested": META_KS,
    "summary": summary,
    "meta_k_selection": meta_k_selection,
    "raw_per_seed_auc": raw_per_seed_auc,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(summary, indent=2))
print("BIC-selected K (OOF-600):", K_STAR, " | (Split-75 no-replace):", K_STAR_SPLIT75)
print("saved ->", OUT)
