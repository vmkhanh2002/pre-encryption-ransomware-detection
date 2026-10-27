"""White-box, mode-targeted attacks on the RanSMAP (I/O) channel.

Level-3 attacker of the paper (Table 4, rows 3a/3b): knows the fitted GMM's
parameters. All attacks act in feature space on the continuous hypervisor
I/O metrics and target per-sample benign structure:

  a. nearest-mode attack: for each malware x, pick the benign GMM component
     k* with minimum Mahalanobis distance to its centre; move
     x -> x + a*(mu_k* - x), with the smallest a such that NLL <= tau
     (tau = median benign training NLL).
  b. responsibility-weighted centroid attack: target t = sum_k r_k(x) mu_k
     (soft convex combination of centres); same line search.
  c. attack cost: minimal displacement a* ||mu_k* - x|| along the
     nearest-mode direction to reach the tau-isodensity surface, per family.
  d. margin distribution: NLL(x) - tau quantiles per family; fraction of
     malware already below tau.
  e. per-process-type imitation: move toward the centroid of the benign
     process type that family F most resembles (largest mean responsibility)
     until NLL <= tau.

Functionality-cost proxies: L-inf relative change and the fraction of
features moved by more than 25%. Post-attack AUC is computed on the attacked
test set (benign samples untouched).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RESULTS_DIR, SEEDS, K_RANSMAP, COV_RANSMAP, REG_COVAR, N_INIT  # noqa: E402
import os, sys, json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from utils import load_ransmap

OUT = RESULTS_DIR / 'whitebox_mode_attack.json'
df, feat_cols = load_ransmap()
X = df[feat_cols].to_numpy(dtype=np.float64)
fam = df['family'].to_numpy()
is_benign = df['label'].to_numpy() == 0
CANONICAL = ['Conti', 'Ryuk', 'LockBit', 'REvil', 'WannaCry']
PTYPES = ['Firefox', 'Idle', 'Office', 'SDelete', 'Zip']

def nll(gm, Xq):
    return -gm.score_samples(Xq)

def solve_alpha(gm, X0, T, tau, max_a=1.0):
    """Per-sample smallest a with NLL(x+a(T-x)) <= tau (bisection on [0,max_a])."""
    def f(a):
        aa = np.asarray(a)
        if aa.ndim == 0:
            return nll(gm, X0 + float(aa) * (T - X0))
        return nll(gm, X0 + aa[:, None] * (T - X0))
    n = len(X0)
    alphas = np.zeros(n)
    inside = f(0.0) <= tau
    reachable = f(max_a) <= tau
    todo = (~inside) & reachable
    lo = np.zeros(n); hi = np.full(n, max_a)
    for _ in range(30):
        mid = (lo + hi) / 2
        ok = f(mid) <= tau
        hi = np.where(ok & todo, mid, hi)
        lo = np.where(~ok & todo, mid, lo)
    alphas[todo] = hi[todo]
    success = inside | reachable
    return alphas, bool(success.all()), success

res = {'description': __doc__.split('\n')[0],
       'per_seed': {}, 'summary': {}}

agg = {k: [] for k in ['clean_auc', 'mean_auc', 'mode_auc', 'rwcent_auc', 'ptype_auc',
                       'mode_success', 'rwcent_success', 'ptype_success',
                       'mode_alpha', 'ptype_alpha', 'mode_linf', 'ptype_linf']}
cost_by_family = {}
margin_by_family = {}

for seed in SEEDS:
    idx_b = np.where(is_benign)[0]
    tr_b, te_b = train_test_split(idx_b, test_size=0.2, random_state=seed)
    gm = GaussianMixture(K_RANSMAP, covariance_type=COV_RANSMAP, reg_covar=REG_COVAR,
                         n_init=N_INIT, random_state=seed).fit(X[tr_b])
    mus = gm.means_
    covs = gm.covariances_
    prec_chol = gm.precisions_cholesky_

    s_tr = nll(gm, X[tr_b])
    tau = float(np.median(s_tr))

    mal_idx = np.array([i for i in np.where(~is_benign)[0] if fam[i] in CANONICAL])
    Xte = np.vstack([X[te_b], X[mal_idx]])
    yte = np.concatenate([np.zeros(len(te_b)), np.ones(len(mal_idx))])
    Xmal = X[mal_idx]

    # ---- (d) margin distribution + per-family cost/margin (seed-averaged later)
    s_mal = nll(gm, Xmal)
    margins = s_mal - tau
    for i, gi in enumerate(mal_idx):
        margin_by_family.setdefault(fam[gi], []).append(float(margins[i]))

    # nearest component by Mahalanobis distance
    def mahal_to_centers(Xq):
        D = np.empty((len(Xq), len(mus)))
        for k in range(len(mus)):
            diff = Xq - mus[k]
            D[:, k] = np.einsum('ij,jk,ik->i', diff, covs[k], diff)
        return D
    kstar = mahal_to_centers(Xmal).argmin(axis=1)
    T_mode = mus[kstar]

    # responsibility-weighted centroid
    resp = gm.predict_proba(Xmal)
    T_rw = resp @ mus

    # (e) per-family best-matching benign process type (by mean responsibility
    #     of family samples over benign-train-fit model), target = subset centroid
    ptype_centroids = {p: X[tr_b][fam[tr_b] == p].mean(axis=0) for p in PTYPES}
    T_ptype = np.empty_like(Xmal)
    for i, gi in enumerate(mal_idx):
        sub = fam == fam[gi]
        r = gm.predict_proba(X[sub]).mean(axis=0)
        # restrict to responsibilities over components dominated by each ptype
        scores_p = {}
        for p in PTYPES:
            cp = ptype_centroids[p]
            d = cp - mus                       # which components sit near this ptype
            w = np.exp(-np.sum((d / np.sqrt(
                np.array([np.diag(c) for c in covs])))**2, axis=1))
            scores_p[p] = float((r * w / (w.sum() + 1e-12)).sum())
        T_ptype[i] = ptype_centroids[max(scores_p, key=scores_p.get)]
    ptype_choice = None  # (diagnostic; per-family target choice is implicit above)

    a_clean = roc_auc_score(yte, nll(gm, Xte))
    out = {'tau_benign_median_nll': tau, 'clean_auc': float(a_clean)}

    # ---- attacks: alpha search per sample (vectorised per group is fine at n~750)
    results_attacks = {}
    for name, T in [('nearest_mode', T_mode), ('resp_weighted_centroid', T_rw),
                    ('process_type', T_ptype)]:
        alphas, all_ok, success_mask = solve_alpha(gm, Xmal, T, tau)
        if success_mask.any():
            Xadv_mal = Xmal + alphas[:, None] * (T - Xmal)
            Xadv = np.vstack([X[te_b], Xadv_mal])
            auc_adv = roc_auc_score(yte, nll(gm, Xadv))
            disp = np.linalg.norm(alphas[:, None] * (T - Xmal), axis=1)
            base = np.maximum(np.abs(T).max(), 1e-9)
            rel = alphas[:, None] * np.abs(T - Xmal) / np.maximum(np.abs(Xmal), 1e-9)
            rec = {'success_rate': float(success_mask.mean()),
                   'post_attack_auc': float(auc_adv),
                   'mean_alpha': float(alphas.mean()),
                   'median_displacement_l2': float(np.median(disp)),
                   'frac_features_gt25pct_rel_change':
                        float(np.mean((rel > 0.25).any(axis=1))),
                   'linf_rel_change_median': float(np.median(rel.max(axis=1)))}
            if name == 'nearest_mode':
                for i, gi in enumerate(mal_idx):
                    cost_by_family.setdefault(fam[gi], []).append(float(disp[i]))
                # per-family post-attack detection rate at tau threshold
                det = (nll(gm, Xadv_mal) > tau).mean()
                rec['detected_frac_at_tau'] = float(det)
        else:
            rec = {'success_rate': 0.0, 'post_attack_auc': None}
        results_attacks[name] = rec

    # global-mean reference attack (alpha=1, the failed baseline)
    mu_glob = X[tr_b].mean(axis=0)
    Xg_mal = mu_glob[None, :] * np.ones_like(Xmal)
    out['globalmean_auc'] = float(roc_auc_score(
        yte, nll(gm, np.vstack([X[te_b], Xg_mal]))))
    out.update(results_attacks)
    res['per_seed'][f'seed{seed}'] = out

    agg['clean_auc'].append(out['clean_auc'])
    agg['mode_auc'].append(results_attacks['nearest_mode']['post_attack_auc'])
    agg['rwcent_auc'].append(results_attacks['resp_weighted_centroid']['post_attack_auc'])
    agg['ptype_auc'].append(results_attacks['process_type']['post_attack_auc'])
    agg['mode_success'].append(results_attacks['nearest_mode']['success_rate'])
    agg['rwcent_success'].append(results_attacks['resp_weighted_centroid']['success_rate'])
    agg['ptype_success'].append(results_attacks['process_type']['success_rate'])
    agg['mode_alpha'].append(results_attacks['nearest_mode']['mean_alpha'])
    agg['ptype_alpha'].append(results_attacks['process_type']['mean_alpha'])
    agg['mode_linf'].append(results_attacks['nearest_mode']['linf_rel_change_median'])
    agg['ptype_linf'].append(results_attacks['process_type']['linf_rel_change_median'])
    print(f"seed {seed}: clean={out['clean_auc']:.4f} "
          f"mode={results_attacks['nearest_mode']['post_attack_auc']:.3f} "
          f"(succ={results_attacks['nearest_mode']['success_rate']:.2f}) "
          f"ptype={results_attacks['process_type']['post_attack_auc']:.3f}", flush=True)

res['summary'] = {
    'clean_auc_mean': float(np.mean(agg['clean_auc'])),
    'nearest_mode_attack': {
        'post_attack_auc_mean': float(np.nanmean(agg['mode_auc'])),
        'evasion_success_mean': float(np.mean(agg['mode_success']))},
    'resp_weighted_centroid_attack': {
        'post_attack_auc_mean': float(np.nanmean(agg['rwcent_auc'])),
        'evasion_success_mean': float(np.mean(agg['rwcent_success']))},
    'process_type_attack': {
        'post_attack_auc_mean': float(np.nanmean(agg['ptype_auc'])),
        'evasion_success_mean': float(np.mean(agg['ptype_success']))},
    'nearest_mode_cost': {
        'alpha_mean': float(np.mean(agg['mode_alpha'])),
        'linf_rel_change_median_mean': float(np.mean(agg['mode_linf'])),
        'linf_rel_change_median_std': float(np.std(agg['mode_linf']))},
    'process_type_cost': {
        'alpha_mean': float(np.mean(agg['ptype_alpha'])),
        'linf_rel_change_median_mean': float(np.mean(agg['ptype_linf'])),
        'linf_rel_change_median_std': float(np.std(agg['ptype_linf']))},
}
res['attack_cost_by_family_nearestmode'] = {
    f: {'median_disp': float(np.median(v)), 'q25': float(np.percentile(v, 25)),
        'q75': float(np.percentile(v, 75))}
    for f, v in sorted(cost_by_family.items())}
res['nll_margin_above_tau_by_family'] = {
    f: {'median_margin': float(np.median(v)),
        'frac_already_below_tau': float(np.mean(np.array(v) < 0))}
    for f, v in sorted(margin_by_family.items())}

with open(OUT, 'w') as f:
    json.dump(res, f, indent=1)
print(json.dumps(res['summary'], indent=1))
print(json.dumps(res['attack_cost_by_family_nearestmode'], indent=1))
print('saved ->', OUT)
