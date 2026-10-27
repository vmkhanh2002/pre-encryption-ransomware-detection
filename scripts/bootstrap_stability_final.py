"""Bootstrap assignment stability for choosing K on the MLRan channel.

Methodology: Hennig (2007) / Lange et al. (2003) cluster stability.
For each K in 1..10:
  - fit a diagonal GMM on the full benign training split -> labels_full
  - for b in 1..20: draw a bootstrap resample, refit GMM(K), predict labels
    on the full training split, compute ARI(labels_full, labels_boot)
  - stability[K] = mean(ARI)
Selection rule: the largest K with mean ARI >= 0.9 * max(stability).

Downstream detection AUC per K is recorded as a diagnostic only, never as a
selection criterion (a benign-only detector must not choose K with labels).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import REG_COVAR, RESULTS_DIR, SEEDS, TEST_SIZE  # noqa: E402
import os, sys, hashlib, json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from utils import load_mlran

N_INIT_REFERENCE = 3  # EM restarts for the reference fit per K
N_INIT_BOOTSTRAP = 2  # EM restarts per bootstrap refit (20 per K and seed)

OUT = RESULTS_DIR / 'bootstrap_stability_final.json'

KS = list(range(1, 11))
N_BOOT = 20
df, feat_cols = load_mlran()
X_all = df[feat_cols].to_numpy(np.float32)
y_all = df['label'].to_numpy()

per_seed = []
for seed in SEEDS:
    # benign-only group-based split (same protocol as paper)
    ghash = df[fm] if False else None
    groups = df[feat_cols].astype(str).agg('|'.join, axis=1).apply(
        lambda s: hashlib.md5(s.encode()).hexdigest())
    tr_g, te_g = train_test_split(groups.unique(), test_size=TEST_SIZE, random_state=seed)
    mtr = groups.isin(set(tr_g)).to_numpy()
    Xb = X_all[mtr & (y_all == 0)]
    Xte = X_all[~mtr]
    yte = y_all[~mtr]

    rec = {'seed': seed, 'stability': {}, 'auc': {}}
    for K in KS:
        try:
            gm_full = GaussianMixture(n_components=K, covariance_type='diag',
                                      reg_covar=REG_COVAR, n_init=N_INIT_REFERENCE,
                                      random_state=seed).fit(Xb)
            lab_full = gm_full.predict(Xb)
        except ValueError:
            rec['stability'][K] = None
            continue
        rng = np.random.RandomState(seed * 100 + K)
        aris = []
        for _ in range(N_BOOT):
            idx = rng.randint(0, len(Xb), size=len(Xb))
            sample = Xb[idx]
            try:
                gm_b = GaussianMixture(n_components=K, covariance_type='diag',
                                       reg_covar=REG_COVAR, n_init=N_INIT_BOOTSTRAP,
                                       random_state=rng.randint(1 << 30)).fit(sample)
                lab_b = gm_b.predict(Xb)
                aris.append(adjusted_rand_score(lab_full, lab_b))
            except ValueError:
                continue
        rec['stability'][K] = round(float(np.mean(aris)), 4)
        # diagnostic downstream AUC
        try:
            auc = roc_auc_score(yte, -gm_full.score_samples(Xte))
            rec['auc'][K] = round(float(auc), 4)
        except Exception:
            pass
    per_seed.append(rec)
    print(f"seed{seed}: " + " ".join(
        f"K{k}={rec['stability'].get(k)}" for k in KS if k in rec['stability']), flush=True)

# aggregate
mean_stab = {K: round(float(np.mean([p['stability'].get(K) or np.nan
                                     for p in per_seed if p['stability'].get(K) is not None])), 4)
             for K in KS}
mean_auc = {K: round(float(np.mean([p['auc'].get(K) for p in per_seed if K in p['auc']])), 4)
            for K in KS}

valid = {k: v for k, v in mean_stab.items() if not np.isnan(v)}
max_stab = max(valid.values())
threshold = 0.9 * max_stab
selected_K = max(k for k, v in valid.items() if v >= threshold)

summary = {
    'method': ('Bootstrap assignment stability (Hennig 2007; Lange et al. 2003). '
               f'{N_BOOT} bootstraps per K. Select largest K with mean ARI >= '
               '90% of max stability.'),
    'mean_ari_per_K': mean_stab,
    'diagnostic_auc_per_K': mean_auc,
    'max_stability': round(max_stab, 4),
    'threshold': round(threshold, 4),
    'selected_K': selected_K,
    'paper_K': 3,
    'matches_paper': selected_K == 3,
}
with open(OUT, 'w') as f:
    json.dump({'summary': summary, 'per_seed': per_seed}, f, indent=1)
print(json.dumps(summary, indent=1))
print('Saved ->', OUT)
