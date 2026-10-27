"""Family-blocked significance test: fused system vs. RanSMAP alone.

Sign-flip permutation test with the canonical family as the blocking unit
(one mean AUC difference per family, averaged over seeds), comparing the
learned fusion with the best single channel. With n families there are only
2^n sign patterns, so the test enumerates all of them and the p-value is
exact; its smallest attainable value is 2 / 2^n (0.0625 for 5 families),
reached when every family favours the same side.

Both calibration variants of learned_fusion.py are tested; the paper's
Table 3 reports the `split75` variant.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RESULTS_DIR, fused_result_key, load_result  # noqa: E402
import itertools
import json
import numpy as np

OUT = RESULTS_DIR / "learned_fusion_significance.json"


def blocked_test(a_folds, b_folds):
    fams = sorted(a_folds)
    d = np.array([np.mean(a_folds[f]) - np.mean(b_folds[f]) for f in fams])
    obs = float(d.mean())
    signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    null = (signs * d).mean(axis=1)
    p = float(np.mean(np.abs(null) >= abs(obs) - 1e-15))
    return {
        "obs_diff": round(obs, 5),
        "per_family": {f: round(float(x), 5) for f, x in zip(fams, d)},
        "n_families_favouring_fused": int((d > 0).sum()),
        "p_value": round(p, 4),
        "min_attainable_p": round(2 / 2 ** len(d), 4),
        "n_sign_patterns": int(len(signs)),
    }


def main():
    fusion = load_result("learned_fusion.json")
    raw = fusion["raw_per_seed_auc"]
    out = {"description": __doc__.split("\n")[0], "tests": {}}
    for variant in ("split75", "oof"):
        key = fused_result_key(variant)
        out["tests"][variant] = {"fused_key": key, **blocked_test(raw[key], raw["ransmap_alone"])}
    out["paper_variant"] = "split75"
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
