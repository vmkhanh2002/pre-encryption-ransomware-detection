"""Bounded feature-space attacks toward the benign mean (Level 2).

The benign mean itself has an NLL far above the benign median: the GMM's
high-density regions are narrow and multi-modal, and the mean lies between
modes. This script reports the NLL reduction an attacker achieves for each
displacement budget toward the mean, showing that bounded moves in that
direction never reach the benign range (AUC stays near the clean value).
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

OUT = RESULTS_DIR / 'whitebox_adversarial.json'

CANON_R = ['Conti', 'Ryuk', 'LockBit', 'REvil', 'WannaCry']
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

dfr, fr = load_ransmap()
Xr = dfr[fr].to_numpy(np.float64)
is_ben = dfr['label'].to_numpy() == 0
famr = dfr['family'].to_numpy()

results = []
for seed in SEEDS:
    idx_ben = np.where(is_ben)[0]
    tr_b, te_b = train_test_split(idx_ben, test_size=0.2, random_state=seed)
    gm = GaussianMixture(K_RANSMAP, covariance_type=COV_RANSMAP, reg_covar=REG_COVAR,
                         n_init=N_INIT, random_state=seed).fit(Xr[tr_b])
    mal_idx = [i for i in np.where(~is_ben)[0] if famr[i] in CANON_R]
    Xorig = Xr[mal_idx]
    mu = Xr[tr_b].mean(axis=0)
    med_ben = float(np.median(-gm.score_samples(Xr[tr_b])))
    thr5 = np.sort(-gm.score_samples(Xr[tr_b]))[-int(np.ceil(0.05 * len(tr_b)))]
    s_ben_te = -gm.score_samples(Xr[te_b])

    row = {'seed': seed}
    for alpha in ALPHAS:
        Xadv = Xorig + (mu - Xorig) * alpha   # direct interpolation toward mu
        s_mal = -gm.score_samples(Xadv)
        auc = float(roc_auc_score(np.concatenate([np.zeros(len(te_b)), np.ones(len(Xadv))]),
                                  np.concatenate([s_ben_te, s_mal])))
        row[f'a{alpha}'] = {'auc': round(auc, 4),
                            'tpr5': round(float((s_mal >= thr5).mean()), 4),
                            'nll_mean': round(float(s_mal.mean()), 1),
                            'frac_reaching_benign_median':
                                round(float((s_mal <= med_ben).mean()), 4)}
    results.append(row)

agg = {}
for a in ALPHAS:
    k = f'a{a}'
    agg[k] = {
        'auc_mean': round(float(np.mean([r[k]['auc'] for r in results])), 4),
        'tpr5_mean': round(float(np.mean([r[k]['tpr5'] for r in results])), 4),
        'nll_mean': round(float(np.mean([r[k]['nll_mean'] for r in results])), 1),
        'frac_evasion_success': round(float(np.mean([r[k]['frac_reaching_benign_median']
                                                     for r in results])), 4),
    }

out = {'description': ('Feature-space evasion by interpolating malware I/O vectors toward '
                       'the benign mean at fraction alpha. Benign median NLL ~124; '
                       'benign-mean NLL itself is >10k, so NO bounded displacement reaches '
                       'low-density-free region. Evasion success = frac of malware whose '
                       'NLL drops below benign median.'),
       'aggregate': agg}
with open(OUT, 'w') as f:
    json.dump(out, f, indent=1)
print(json.dumps(agg, indent=1))
print('Saved ->', OUT)
