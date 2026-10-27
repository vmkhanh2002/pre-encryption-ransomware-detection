"""Worst-case adversarial search against the fused system, v1 (static heuristic).

Two per-sample attacks that target the fused meta-GMM score directly,
rather than each channel's own threshold (as the Level-3 attack in
fused_under_attack_learned.py does):

  Attack A: fusion-aware single-channel suppression (RanSMAP only).
    For each malware sample, grid-search the blend fraction alpha in
    linspace(0, 1, 41) toward the nearest benign GMM mode (same target as
    whitebox_mode_attack.py) and keep the alpha that minimizes the fused
    score. MLRan is left untouched.

  Attack B: joint two-channel attack. MLRan's 482 features are strictly
    binary, so continuous interpolation does not apply; a greedy 1->0
    bit-flip heuristic suppresses indicators instead (0->1 decoy insertion is
    out of scope). Under the sample's nearest MLRan GMM component k*
    (diagonal covariance, so the log-density factorises per dimension given
    k*), set bits are ranked by the NLL reduction of flipping them to 0, and
    the top-m are flipped for m in MLRAN_BIT_BUDGETS. The ranking is computed
    once (k* is not re-derived after each flip); the true GMM NLL scores each
    perturbed vector. A 2-D grid over (MLRan bit budget) x (RanSMAP alpha) is
    then searched per sample for the minimum fused score. This is a grid
    search over independent per-channel budgets, not a joint optimisation.

Both attacks reuse the leak-free calibration/evaluation split of
learned_fusion.py. joint_channel_adversarial_v2.py replaces the static
search with coordinate descent.
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
OUT = RESULTS_DIR / 'joint_channel_adversarial.json'
CANONICAL = ['Conti', 'Ryuk', 'LockBit', 'REvil', 'WannaCry']
MLRAN_EXCLUDE = {"label", "family", "sample_id", "sha256", "split", "STRING:labels", "dataset"}
MLRAN_FAMILY_MAP = {"Conti": "conti", "Ryuk": "ryuk", "LockBit": "lockbit",
                     "REvil": "revil", "WannaCry": "wannacry"}
META_K = selected_meta_k()

MLRAN_BIT_BUDGETS = [0, 5, 10, 20, 40, 80, 160, 242, 482]
RANSMAP_ALPHAS = np.linspace(0, 1, 41)

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

# Confirm MLRan's binary nature once at import time (the design premise of Attack B).
_nunique = mlran[mfc].nunique()
assert (_nunique <= 2).all(), "MLRan features assumed binary: found a non-binary column"


def nll(gm, Xq):
    return -gm.score_samples(Xq)


def pair(a, b, rng, n):
    ia = rng.choice(len(a), n, replace=(len(a) < n))
    ib = rng.choice(len(b), n, replace=(len(b) < n))
    return a[ia], b[ib]


def split_half(arr, rng):
    idx = rng.permutation(len(arr))
    h = len(arr) // 2
    return arr[idx[:h]], arr[idx[h:]]


def mahal_to_centers_full(Xq, mus, covs):
    D = np.empty((len(Xq), len(mus)))
    for k in range(len(mus)):
        diff = Xq - mus[k]
        D[:, k] = np.einsum('ij,jk,ik->i', diff, covs[k], diff)
    return D


def rank_flip_order(x, mu_k, var_k):
    """Indices of x's 1-valued bits, ranked by NLL reduction if flipped to 0
    under a FIXED diag-covariance component (mu_k, var_k), most-reducing
    first. Approximation: does not re-derive the nearest component after
    each flip."""
    ones = np.where(x == 1)[0]
    if len(ones) == 0:
        return ones
    term_before = (x[ones] - mu_k[ones]) ** 2 / var_k[ones]
    term_after = (0.0 - mu_k[ones]) ** 2 / var_k[ones]
    delta = term_before - term_after   # positive => flipping to 0 reduces NLL contribution
    return ones[np.argsort(-delta)]


def mlran_budget_grid_scores(gm_m, sm_tr, Xmmal):
    """Returns (n_samples, n_budgets) MinMax-normalised NLL after flipping the
    top-`budget` ranked bits per sample, for budget in MLRAN_BIT_BUDGETS."""
    mus_m, vars_m = gm_m.means_, gm_m.covariances_   # diag: shape (K, D)
    d = np.stack([np.sum((Xmmal - mus_m[k]) ** 2 / vars_m[k], axis=1)
                  for k in range(K_MLRAN)], axis=1)
    kstar = d.argmin(axis=1)

    n, D = Xmmal.shape
    n_budgets = len(MLRAN_BIT_BUDGETS)
    Xpert = np.repeat(Xmmal[:, None, :], n_budgets, axis=1).copy()  # (n, budgets, D)
    for i in range(n):
        order = rank_flip_order(Xmmal[i], mus_m[kstar[i]], vars_m[kstar[i]])
        if len(order) == 0:
            continue
        for bi, budget in enumerate(MLRAN_BIT_BUDGETS):
            if budget == 0:
                continue
            flips = order[:min(budget, len(order))]
            Xpert[i, bi, flips] = 0.0

    raw = nll(gm_m, Xpert.reshape(-1, D)).reshape(n, n_budgets)
    return minmax_norm(sm_tr, raw.reshape(-1)).reshape(n, n_budgets)


def ransmap_alpha_grid_scores(gm_r, sr_tr, Xrmal_f, T_mode_f):
    """Returns (n_samples, n_alphas) MinMax-normalised NLL as each sample
    moves toward its nearest-mode target T_mode_f, for alpha in RANSMAP_ALPHAS."""
    n, D = Xrmal_f.shape
    ts = RANSMAP_ALPHAS
    Xmoves = Xrmal_f[:, None, :] + (T_mode_f - Xrmal_f)[:, None, :] * ts[None, :, None]
    raw = nll(gm_r, Xmoves.reshape(-1, D)).reshape(n, len(ts))
    return minmax_norm(sr_tr, raw.reshape(-1)).reshape(n, len(ts))


res = {'description': __doc__.strip().split('\n')[0],
       'mlran_bit_budgets': MLRAN_BIT_BUDGETS,
       'ransmap_alphas_tested': len(RANSMAP_ALPHAS),
       'per_seed': {}}
agg = {'clean': [], 'attackA_ransmap_only': [], 'attackB_joint_worst_case': []}
agg_per_budget = {b: [] for b in MLRAN_BIT_BUDGETS}

for seed in SEEDS:
    rng_pair, rng_split, rng_meta, rng_join = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4))

    # MLRan channel (clean fit, matches ano_combined_eval.py)
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
    mus_r, covs_r = gm_r.means_, gm_r.covariances_

    mal_idx = np.array([i for i in np.where(~is_benign_r)[0] if fam_r[i] in CANONICAL])
    Xrmal = Xr[mal_idx]
    fam_mal = fam_r[mal_idx]
    kstar_r = mahal_to_centers_full(Xrmal, mus_r, covs_r).argmin(axis=1)
    T_mode = mus_r[kstar_r]

    sm_ben_n_full = minmax_norm(sm_tr, nll(gm_m, m_ben[~m_tr_mask][mfc].fillna(0.0).values))
    sr_ben_n_full = minmax_norm(sr_tr, nll(gm_r, Xr[te_b]))

    # leak-free calib/eval split of benign TEST scores, per channel
    sm_ben_calib, sm_ben_eval = split_half(sm_ben_n_full, rng_split)
    sr_ben_calib, sr_ben_eval = split_half(sr_ben_n_full, rng_split)

    n_calib_p = min(len(sm_ben_calib), len(sr_ben_calib))
    pa_c, pb_c = pair(sm_ben_calib, sr_ben_calib, rng_meta, max(n_calib_p, 200))
    meta = GaussianMixture(META_K, covariance_type='full', reg_covar=META_REG_COVAR,
                           n_init=N_INIT, random_state=seed).fit(np.column_stack([pa_c, pb_c]))

    n_eval_p = min(len(sm_ben_eval), len(sr_ben_eval))

    seed_out = {}
    for family in CANONICAL:
        Xmmal = m_mal[family][mfc].fillna(0.0).values
        sm_mal_grid = mlran_budget_grid_scores(gm_m, sm_tr, Xmmal)          # (n_m, n_budgets)

        idxf = np.where(fam_mal == family)[0]
        sr_mal_grid = ransmap_alpha_grid_scores(gm_r, sr_tr, Xrmal[idxf], T_mode[idxf])  # (n_r, n_alphas)

        n_pair = min(len(sm_mal_grid), len(sr_mal_grid))
        ia = rng_join.choice(len(sm_mal_grid), n_pair, replace=(len(sm_mal_grid) < n_pair))
        ib = rng_join.choice(len(sr_mal_grid), n_pair, replace=(len(sr_mal_grid) < n_pair))
        sm_pair_grid = sm_mal_grid[ia]     # (n_pair, n_budgets)
        sr_pair_grid = sr_mal_grid[ib]     # (n_pair, n_alphas)

        n_budgets, n_alphas = sm_pair_grid.shape[1], sr_pair_grid.shape[1]
        # joint (n_pair, n_budgets, n_alphas) fused-NLL grid
        XY = np.empty((n_pair, n_budgets, n_alphas, 2))
        XY[..., 0] = sm_pair_grid[:, :, None]
        XY[..., 1] = sr_pair_grid[:, None, :]
        fused_grid = (-meta.score_samples(XY.reshape(-1, 2))).reshape(n_pair, n_budgets, n_alphas)

        s_clean = fused_grid[:, 0, 0]                          # budget=0, alpha=0 (unattacked)
        s_atkA = fused_grid[:, 0, :].min(axis=1)                # Attack A: RanSMAP-only, fused-aware
        s_per_budget = fused_grid.min(axis=2)                   # (n_pair, n_budgets): best alpha per budget
        s_atkB_joint = fused_grid.reshape(n_pair, -1).min(axis=1)  # Attack B: joint worst case

        pab, pbb = pair(sm_ben_eval, sr_ben_eval, rng_pair, n_eval_p)
        s_ben_meta = -meta.score_samples(np.column_stack([pab, pbb]))
        y = np.concatenate([np.zeros(n_eval_p), np.ones(n_pair)])

        auc_clean = roc_auc_score(y, np.concatenate([s_ben_meta, s_clean]))
        auc_atkA = roc_auc_score(y, np.concatenate([s_ben_meta, s_atkA]))
        auc_atkB = roc_auc_score(y, np.concatenate([s_ben_meta, s_atkB_joint]))
        auc_per_budget = {
            str(MLRAN_BIT_BUDGETS[bi]): float(roc_auc_score(
                y, np.concatenate([s_ben_meta, s_per_budget[:, bi]])))
            for bi in range(n_budgets)
        }

        seed_out[family] = {
            'auc_clean': float(auc_clean),
            'auc_attackA_ransmap_only': float(auc_atkA),
            'auc_attackB_joint_worst_case': float(auc_atkB),
            'auc_by_mlran_budget': auc_per_budget,
        }
        agg['clean'].append(auc_clean)
        agg['attackA_ransmap_only'].append(auc_atkA)
        agg['attackB_joint_worst_case'].append(auc_atkB)
        for b in MLRAN_BIT_BUDGETS:
            agg_per_budget[b].append(auc_per_budget[str(b)])

    res['per_seed'][f'seed{seed}'] = seed_out
    print(f"seed {seed}: clean={np.mean([seed_out[f]['auc_clean'] for f in CANONICAL]):.4f} "
          f"attackA={np.mean([seed_out[f]['auc_attackA_ransmap_only'] for f in CANONICAL]):.4f} "
          f"attackB={np.mean([seed_out[f]['auc_attackB_joint_worst_case'] for f in CANONICAL]):.4f}",
          flush=True)

res['summary'] = {
    'macro_auc_clean': float(np.mean(agg['clean'])),
    'macro_auc_attackA_ransmap_only': float(np.mean(agg['attackA_ransmap_only'])),
    'macro_auc_attackB_joint_worst_case': float(np.mean(agg['attackB_joint_worst_case'])),
    'macro_auc_by_mlran_budget': {str(b): float(np.mean(v)) for b, v in agg_per_budget.items()},
    'reference_existing_level3_tau_based_attack': {
        'note': ('fused_under_attack_learned.json b_learned_auc_attacked_mean: the '
                 'existing single-channel (RanSMAP-only), tau-threshold-targeted attack, '
                 'kept here only as a cross-reference, not recomputed by this script.'),
    },
}
with open(OUT, 'w') as f:
    json.dump(res, f, indent=1)
print(json.dumps(res['summary'], indent=1))
print('saved ->', OUT)
