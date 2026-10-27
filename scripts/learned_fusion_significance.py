"""Family-blocked significance test: fused system vs. RanSMAP alone.

Sign-flip permutation test with the canonical family as the blocking unit
(one mean AUC difference per family), comparing the learned fusion with the
best single channel. Reads the per-seed, per-family AUC arrays that
learned_fusion.py stores under `raw_per_seed_auc`, for the component count
it selected (`meta_k_selection.selected_k`) (paper, Section 4.2).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RESULTS_DIR  # noqa: E402
import json
from pathlib import Path
import numpy as np

FUSION_JSON = RESULTS_DIR / 'learned_fusion.json'
OUT = RESULTS_DIR / 'learned_fusion_significance.json'


def blocked_test(a_folds, b_folds, n_perm=20000, seed=42):
    """a, b: dict family -> list of fold AUCs. Returns p-value two-sided.

    One mean-AUC-diff per family (the correct unit of exchangeability, since
    per-seed folds within a family are not independent replicates), signs
    flipped under the null.
    """
    fams = sorted(a_folds.keys())
    obs = []
    means_a, means_b = [], []
    for f in fams:
        ma_, mb_ = np.mean(a_folds[f]), np.mean(b_folds[f])
        obs.append(ma_ - mb_)
        means_a.append(ma_); means_b.append(mb_)
    obs_stat = float(np.mean(obs))
    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(fams))
        stat = float(np.mean(np.array(obs) * signs))
        if abs(stat) >= abs(obs_stat):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {'obs_diff': round(obs_stat, 5),
            'per_family': {f: round(o, 5) for f, o in zip(fams, obs)},
            'p_value': round(p, 4)}


def main():
    fusion = json.load(open(FUSION_JSON))
    k_star = fusion['meta_k_selection']['selected_k']
    fused_key = f'meta_K{k_star}'

    b_learned_folds = fusion['raw_per_seed_auc'][fused_key]
    ransmap_folds = fusion['raw_per_seed_auc']['ransmap_alone']

    test = blocked_test(b_learned_folds, ransmap_folds)

    out = {
        'description': (
            'Family-blocked sign-flip permutation test: B_learned '
            f'({fused_key}, BIC-selected) vs. RanSMAP alone, on the identical '
            '5-family/10-seed folds (no additional model fitting, reuses '
            'learned_fusion.json\'s raw_per_seed_auc).'
        ),
        'fused_key_used': fused_key,
        'fold_counts': {
            'b_learned': {f: len(v) for f, v in b_learned_folds.items()},
            'ransmap_alone': {f: len(v) for f, v in ransmap_folds.items()},
        },
        'test': test,
        'family_means': {
            'b_learned': {f: round(float(np.mean(v)), 5) for f, v in b_learned_folds.items()},
            'ransmap_alone': {f: round(float(np.mean(v)), 5) for f, v in ransmap_folds.items()},
        },
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(test, indent=1))
    print(json.dumps(out['family_means'], indent=1))
    print('saved ->', OUT)


if __name__ == '__main__':
    main()
