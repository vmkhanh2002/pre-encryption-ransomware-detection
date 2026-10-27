"""Worst-case adversarial search v2: coordinate descent on the joint fused objective.

Limitations of v1 (joint_channel_adversarial.py) that this version removes:
  - MLRan's greedy bit-flip ranking was computed once per sample under one
    fixed nearest-component assignment and never re-derived as bits were
    flipped, although flipping bits can change the nearest component (the
    mixture NLL is a logsumexp over all components).
  - The "joint optimum" was a grid search over independent per-channel
    budgets (MLRan bit count x RanSMAP blend fraction), not an attacker that
    adapts one channel's move to the other's.
  - As a result, v1's macro AUC was non-monotonic in the MLRan bit budget.

v2 targets the fused meta-GMM score at every step:
  - MLRan: the nearest component is re-derived after every flip, and a flip
    is accepted only if it lowers the fused score.
  - RanSMAP: each round, the blend fraction toward the nearest mode is
    re-optimized by grid search at the current MLRan operating point.
Alternating the two half-steps is coordinate descent on the joint objective
and never increases the fused score.

This is still a heuristic local search, not a certified worst case
(coordinate descent can stop at a local optimum, and the binary MLRan space
is not searched exhaustively). Both v1 and v2 results are reported
(paper, Section 4.2).
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

OUT = RESULTS_DIR / "joint_channel_adversarial_v2.json"
CANONICAL = CANONICAL_FAMILIES
MLRAN_FAMILY_MAP = {f: f.lower() for f in CANONICAL_FAMILIES}
META_K = selected_meta_k()

N_ROUNDS = 60
REPORT_ROUNDS = [0, 5, 10, 20, 40, 60]
RANSMAP_ALPHAS = np.linspace(0, 1, 41)

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


def pair_no_inflate(a, b, rng, n):
    ia = rng.choice(len(a), n, replace=(len(a) < n))
    ib = rng.choice(len(b), n, replace=(len(b) < n))
    return a[ia], b[ib]


def coordinate_descent_joint(
    gm_m, sm_tr, X_m0, gm_r, sr_tr, X_r0, T_mode, meta, n_rounds, report_rounds
):
    n = len(X_m0)
    x_m = X_m0.copy()
    flipped = np.zeros_like(x_m, dtype=bool)
    alpha = np.zeros(n)
    mus_m, vars_m = gm_m.means_, gm_m.covariances_
    K = len(mus_m)
    ts = RANSMAP_ALPHAS

    def fused_score(x_m_now, alpha_now):
        sm = minmax_norm(sm_tr, nll(gm_m, x_m_now))
        Xr_now = X_r0 + alpha_now[:, None] * (T_mode - X_r0)
        sr = minmax_norm(sr_tr, nll(gm_r, Xr_now))
        return -meta.score_samples(np.column_stack([sm, sr])), sm, sr

    checkpoints = {}
    fused_now, sm_now, sr_now = fused_score(x_m, alpha)
    if 0 in report_rounds:
        checkpoints[0] = fused_now.copy()

    for round_i in range(1, n_rounds + 1):
        Xmoves = X_r0[:, None, :] + (T_mode - X_r0)[:, None, :] * ts[None, :, None]
        nll_r_grid = nll(gm_r, Xmoves.reshape(-1, X_r0.shape[1])).reshape(n, len(ts))
        sr_grid = minmax_norm(sr_tr, nll_r_grid.reshape(-1)).reshape(n, len(ts))
        XY = np.empty((n, len(ts), 2))
        XY[:, :, 0] = sm_now[:, None]
        XY[:, :, 1] = sr_grid
        fused_grid = -meta.score_samples(XY.reshape(-1, 2)).reshape(n, len(ts))
        best_idx = fused_grid.argmin(axis=1)
        alpha = ts[best_idx]
        sr_now = sr_grid[np.arange(n), best_idx]
        fused_now = fused_grid[np.arange(n), best_idx]

        d = np.stack([
            np.sum((x_m - mus_m[k]) ** 2 / vars_m[k], axis=1) for k in range(K)
        ], axis=1)
        kstar = d.argmin(axis=1)
        mu_k, var_k = mus_m[kstar], vars_m[kstar]
        is_one = (x_m == 1) & (~flipped)
        delta_local = np.where(
            is_one,
            (x_m - mu_k) ** 2 / var_k - (0.0 - mu_k) ** 2 / var_k,
            -np.inf,
        )
        best_j = delta_local.argmax(axis=1)
        has_candidate = delta_local[np.arange(n), best_j] > -np.inf

        x_trial = x_m.copy()
        x_trial[has_candidate, best_j[has_candidate]] = 0.0
        sm_trial = minmax_norm(sm_tr, nll(gm_m, x_trial))
        fused_trial = -meta.score_samples(np.column_stack([sm_trial, sr_now]))

        improve = has_candidate & (fused_trial < fused_now)
        x_m[improve, best_j[improve]] = 0.0
        flipped[improve, best_j[improve]] = True
        sm_now = np.where(improve, sm_trial, sm_now)
        fused_now = np.where(improve, fused_trial, fused_now)

        if round_i in report_rounds:
            checkpoints[round_i] = fused_now.copy()

    return checkpoints, x_m, alpha


res = {
    "description": __doc__.split("\n")[0],
    "n_rounds": N_ROUNDS,
    "report_rounds": REPORT_ROUNDS,
    "per_seed": {},
}
agg = {r: [] for r in REPORT_ROUNDS}
agg_flips_used = []

for seed in SEEDS:
    rng_pair, rng_split, rng_meta, rng_join = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    )

    tr_gids, _ = train_test_split(m_unique_gids, test_size=TEST_SIZE, random_state=seed)
    m_tr_mask = m_ben["_gid"].isin(set(tr_gids))
    Xmb_tr = m_ben[m_tr_mask][mfc].fillna(0.0).values
    gids_tr = m_ben[m_tr_mask]["_gid"].values
    gm_m = fit_gm(Xmb_tr, n_components=K_MLRAN, covariance_type="diag", seed=seed)
    sm_tr = nll(gm_m, Xmb_tr)

    # stratify + StandardScaler on RanSMAP benign train
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
    mus_r = gm_r.means_

    mal_idx = np.array([i for i in np.where(~is_benign_r)[0] if fam_r[i] in CANONICAL])
    Xrmal = Xr[mal_idx]
    fam_mal = fam_r[mal_idx]
    # Mahalanobis distance using gm_r.precisions_
    kstar_r = mahal_sq_to_centers(Xrmal, gm_r).argmin(axis=1)
    T_mode = mus_r[kstar_r]

    sm_ben_n_full = minmax_norm(
        sm_tr, nll(gm_m, m_ben[~m_tr_mask][mfc].fillna(0.0).values)
    )
    sr_ben_n_full = minmax_norm(sr_tr, nll(gm_r, Xr[te_b]))

    # 5-fold OOF training calibration (n=600) + full 150 test evaluation
    sm_oof = oof_benign_norm_scores_mlran(Xmb_tr, gids_tr, K_MLRAN, seed)
    sr_oof = oof_benign_norm_scores_ransmap(
        Xr_raw[tr_b], fam_r[tr_b], K_RANSMAP, seed, scale=True
    )
    n_calib_p = min(len(sm_oof), len(sr_oof))
    pa_c, pb_c = pair_no_inflate(sm_oof, sr_oof, rng_meta, n_calib_p)
    meta = GaussianMixture(
        META_K, covariance_type="full", reg_covar=META_REG_COVAR, n_init=N_INIT, random_state=seed
    ).fit(np.column_stack([pa_c, pb_c]))

    n_eval_p = min(len(sm_ben_n_full), len(sr_ben_n_full))

    seed_out = {}
    for family in CANONICAL:
        Xmmal = m_mal[family][mfc].fillna(0.0).values
        idxf = np.where(fam_mal == family)[0]
        Xrmal_f = Xrmal[idxf]
        T_mode_f = T_mode[idxf]

        n_pair = min(len(Xmmal), len(Xrmal_f))
        ia = rng_join.choice(len(Xmmal), n_pair, replace=(len(Xmmal) < n_pair))
        ib = rng_join.choice(len(Xrmal_f), n_pair, replace=(len(Xrmal_f) < n_pair))
        X_m0 = Xmmal[ia]
        X_r0 = Xrmal_f[ib]
        T_mode_paired = T_mode_f[ib]

        checkpoints, x_m_final, alpha_final = coordinate_descent_joint(
            gm_m, sm_tr, X_m0, gm_r, sr_tr, X_r0, T_mode_paired, meta, N_ROUNDS, REPORT_ROUNDS
        )

        pab, pbb = pair_no_inflate(sm_ben_n_full, sr_ben_n_full, rng_pair, n_eval_p)
        s_ben_meta = -meta.score_samples(np.column_stack([pab, pbb]))
        y = np.concatenate([np.zeros(n_eval_p), np.ones(n_pair)])

        auc_by_round = {}
        for r in REPORT_ROUNDS:
            s_all = np.concatenate([s_ben_meta, checkpoints[r]])
            auc_by_round[str(r)] = float(roc_auc_score(y, s_all))
            agg[r].append(auc_by_round[str(r)])

        n_flips_used = float(np.mean(x_m_final != X_m0))
        agg_flips_used.append(n_flips_used)
        seed_out[family] = {
            "auc_by_round": auc_by_round,
            "mean_frac_bits_flipped": n_flips_used,
        }

    res["per_seed"][f"seed{seed}"] = seed_out
    print(
        f"seed {seed}: "
        + " | ".join(
            f"r{r}={np.mean([seed_out[f]['auc_by_round'][str(r)] for f in CANONICAL]):.4f}"
            for r in REPORT_ROUNDS
        ),
        flush=True,
    )

res["summary"] = {
    "macro_auc_by_round": {str(r): float(np.mean(v)) for r, v in agg.items()},
    "mean_frac_mlran_bits_flipped_at_final_round": float(np.mean(agg_flips_used)),
}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res["summary"], indent=2))
print("saved ->", OUT)
