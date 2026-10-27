"""Precision (PPV) at realistic ransomware prevalence.

Post-hoc analysis without refitting: reads TPR/FPR operating points from
ano_models_results.json and learned_fusion.json and recomputes precision at
assumed base rates far below the roughly balanced evaluation sets:

PPV(prevalence, tpr, fpr) = (prevalence * tpr) /
                            (prevalence * tpr + (1 - prevalence) * fpr)

As prevalence falls, false positives from the much larger benign population
outnumber true positives, even for a detector with high balanced-set AUC.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RESULTS_DIR  # noqa: E402
import json
from pathlib import Path

OUT = RESULTS_DIR / "prevalence_ppv_analysis.json"

MODELS_JSON = RESULTS_DIR / "ano_models_results.json"
FUSION_JSON = RESULTS_DIR / "learned_fusion.json"

PREVALENCES = [1e-2, 1e-3, 1e-4, 1e-5]
OPERATING_POINTS = ["fpr1", "fpr5", "deploy95"]


def ppv(prevalence: float, tpr: float, fpr: float) -> float:
    denom = prevalence * tpr + (1 - prevalence) * fpr
    if denom <= 0:
        return float("nan")
    return (prevalence * tpr) / denom


def channel_operating_points(mean: dict) -> dict:
    """Extract {op_name: (fpr, tpr)} from an ano_models_results.json
    per-model `mean_*` dict. `fpr1`/`fpr5` use the nominal target FPR
    (0.01/0.05) paired with the interpolated TPR at that target; `deploy95`
    uses the *empirical* FPR realised by the train-percentile threshold,
    since that is the actual field-relevant operating point, not a nominal
    target."""
    return {
        "fpr1":     (0.01, mean["mean_tpr_at_fpr1"]),
        "fpr5":     (0.05, mean["mean_tpr_at_fpr5"]),
        "deploy95": (mean["mean_deploy95_fpr"], mean["mean_deploy95_tpr"]),
    }


def fusion_operating_points(macro: dict) -> dict:
    return {
        "fpr1":     (0.01, macro["macro_tpr_at_fpr1"]),
        "fpr5":     (0.05, macro["macro_tpr_at_fpr5"]),
        "deploy95": (macro["macro_deploy95_fpr"], macro["macro_deploy95_tpr"]),
    }


def main():
    models = json.load(open(MODELS_JSON))
    fusion = json.load(open(FUSION_JSON))

    # `learned_fusion.json`'s reported strategy today is meta_K3 (the "best
    # of K in {1,2,3}" pick this repo's item-1 fix will revisit separately;
    # this script re-reads whichever key is current, so rerunning it after
    # that fix picks up the new K automatically without editing this file).
    fused_key = "meta_K3" if "meta_K3" in fusion["summary"] else next(
        k for k in fusion["summary"] if k.startswith("meta_K")
    )

    channels = {
        "MLRan alone":       channel_operating_points(models["ano_mlran"]["GaussianMixture"]),
        "RanSMAP alone":     channel_operating_points(models["ano_ransmap"]["GaussianMixture"]),
        f"Fused ({fused_key})": fusion_operating_points(fusion["summary"][fused_key]),
    }

    results = {}
    header = f"{'Channel':<22}{'Op. point':<12}{'FPR':>8}{'TPR':>8}" + \
             "".join(f"{'PPV@'+f'{p:g}':>14}" for p in PREVALENCES)
    print(header)
    print("-" * len(header))

    for channel_name, ops in channels.items():
        results[channel_name] = {}
        for op_name in OPERATING_POINTS:
            fpr, tpr = ops[op_name]
            row = {"fpr": fpr, "tpr": tpr,
                   "ppv": {f"{p:g}": ppv(p, tpr, fpr) for p in PREVALENCES}}
            results[channel_name][op_name] = row
            ppv_cells = "".join(f"{row['ppv'][f'{p:g}']:>14.4%}" for p in PREVALENCES)
            print(f"{channel_name:<22}{op_name:<12}{fpr:>8.4f}{tpr:>8.4f}{ppv_cells}")

    # Sanity checks (assert, not just print, catches a formula bug loudly).
    assert abs(ppv(0.5, 0.9, 0.1) - 0.9) < 1e-9, "PPV at prevalence=0.5 (balanced) must reduce to TPR/(TPR+FPR)"
    assert abs(ppv(1.0, 0.9, 0.1) - 1.0) < 1e-9, "PPV at prevalence=1 (all positive) must be 1"
    # PPV must be monotonically non-decreasing as prevalence rises, for a fixed op point.
    for channel_ops in results.values():
        for op in channel_ops.values():
            vals = [op["ppv"][f"{p:g}"] for p in sorted(PREVALENCES)]
            assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])), \
                "PPV must increase monotonically with prevalence"

    out = {
        "description": __doc__.strip().split("\n")[0],
        "prevalences_tested": PREVALENCES,
        "fused_key_used": fused_key,
        "formula": "ppv = (prevalence*tpr) / (prevalence*tpr + (1-prevalence)*fpr)",
        "note": (
            "fpr1/fpr5 use the nominal target FPR (0.01/0.05) with the "
            "interpolated TPR at that target; deploy95 uses the *empirical* "
            "FPR realised by the train-percentile-95 threshold on the test "
            "set (mean_deploy95_fpr / macro_deploy95_fpr), which is the "
            "field-relevant operating point since it is not forced to hit "
            "exactly 5% FPR."
        ),
        "results": results,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print()
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
