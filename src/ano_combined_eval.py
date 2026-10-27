"""
LOFO combined anomaly evaluation on 5 canonical families.
5 folds x 10 seeds per option.

Option A - Independent per-channel + probabilistic union estimate
  Train each channel on benign only. Score independently. Union TPR from both
  ROC curves: P(detected) = 1 − (1−TPR_mlran)(1−TPR_ransmap).

Option B - Bootstrap score fusion (5 strategies)
  Normalize both channel scores (MinMax on benign train scores).
  Bootstrap-pair samples across channels (different machines → no natural alignment).
  Fuse: mean | weighted_mean (by per-channel AUC) | max (OR) | min (AND) | product.

Option C - Feature concatenation (single GaussianMixture)
  Concatenate MLRan (482) + RanSMAP (34) = 516 features per paired sample.
  Train single GM(K=1, diag covariance) on bootstrap-paired benign features.
  Evaluate on bootstrap-paired test set.

Metrics per fold/seed/strategy:
  AUC, AUPRC, F1, Precision, Recall, MCC (at Youden's J threshold),
  TPR@FPR=1%, TPR@FPR=5%.
"""

from config import MLRAN_CSV, RANSMAP_CSV, RESULTS_DIR, SEEDS, CANONICAL_FAMILIES, K_MLRAN, K_RANSMAP, load_result  # noqa: E402
import hashlib
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

# Shared, single-source-of-truth implementations (CRIT-01, CRIT-03).
# These are behaviour-identical to the previous local copies, so reported
# numbers are unchanged.
from utils import fit_gm, minmax_norm

# by BIC-selected on the calibration half; see learned_fusion.py)

MLRAN_PATH   = MLRAN_CSV
RANSMAP_PATH = RANSMAP_CSV
OUTPUT_DIR   = RESULTS_DIR

MLRAN_EXCLUDE   = {"label", "family", "sample_id", "sha256", "split", "STRING:labels", "dataset"}
RANSMAP_EXCLUDE = {"label", "family", "trace_path", "dataset"}

MLRAN_FAMILY_MAP  = {"Conti": "conti",  "Ryuk": "ryuk",  "LockBit": "lockbit",
                      "REvil": "revil",  "WannaCry": "wannacry"}
RANSMAP_FAMILY_MAP = {"Conti": "Conti", "Ryuk": "Ryuk",  "LockBit": "LockBit",
                      "REvil": "REvil", "WannaCry": "WannaCry"}

# MAJ-01: B_weighted_mean weights are fixed design constants, set proportional to
# the standalone per-channel AUCs (group-split MLRan K=3, stratified RanSMAP K=5).
# Read from ano_models_eval.py's output (run it first), so the weights always
# match the reported per-channel AUCs.
_PER_CHANNEL = load_result("ano_models_results.json")
_AUC_MLRAN = _PER_CHANNEL["ano_mlran"]["GaussianMixture"]["mean_auc"]
_AUC_RANSMAP = _PER_CHANNEL["ano_ransmap"]["GaussianMixture"]["mean_auc"]
_W_TOTAL   = _AUC_MLRAN + _AUC_RANSMAP
W_MLRAN    = _AUC_MLRAN / _W_TOTAL
W_RANSMAP  = _AUC_RANSMAP / _W_TOTAL
assert abs(W_MLRAN + W_RANSMAP - 1.0) < 1e-9, "fusion weights must sum to 1"

FUSION_STRATEGIES = ["mean", "weighted_mean", "max", "min", "product"]


# --- helpers ------------------------------------------------------------------

def score(gm: GaussianMixture, X: np.ndarray) -> np.ndarray:
    return -gm.score_samples(X)   # higher = more anomalous


def compute_metrics(y: np.ndarray, s: np.ndarray, keep_curve: bool = False) -> dict:
    fpr, tpr, thr = roc_curve(y, s)
    auc   = float(roc_auc_score(y, s))
    auprc = float(average_precision_score(y, s))

    j     = int(np.argmax(tpr - fpr))
    y_hat = (s >= thr[j]).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, y_hat, average="binary", zero_division=0,
    )
    mcc = float(matthews_corrcoef(y, y_hat))

    def _tpr_at(target_fpr):
        return float(np.interp(target_fpr, fpr, tpr))

    out = {
        "auc": auc, "auprc": auprc,
        "f1_opt": float(f1), "precision_opt": float(pr), "recall_opt": float(rc),
        "mcc_opt": mcc,
        "tpr_at_fpr1": _tpr_at(0.01), "tpr_at_fpr5": _tpr_at(0.05),
    }
    if keep_curve:
        out["_fpr"] = fpr.tolist()
        out["_tpr"] = tpr.tolist()
    return out


def union_metrics(m_a: dict, m_b: dict) -> dict:
    """Option A: probabilistic union from both ROC curves."""
    grid  = np.linspace(0, 1, 2000)
    tpr_a = np.interp(grid, m_a["_fpr"], m_a["_tpr"])
    tpr_b = np.interp(grid, m_b["_fpr"], m_b["_tpr"])
    tpr_u = 1.0 - (1.0 - tpr_a) * (1.0 - tpr_b)
    auc_u = float(np.trapezoid(tpr_u, grid))

    def _tpr_at(fpr_val):
        return float(np.interp(fpr_val, grid, tpr_u))

    # Theoretical F1 at Youden-J point on union curve
    j      = int(np.argmax(tpr_u - grid))
    tpr_j  = float(tpr_u[j])
    fpr_j  = float(grid[j])
    # Use majority-vote sample sizes for F1 approximation
    n_ben  = max(m_a.get("_n_ben", 1), m_b.get("_n_ben", 1))
    n_mal  = min(m_a.get("_n_mal", 1), m_b.get("_n_mal", 1))
    tp     = tpr_j * n_mal
    fp     = fpr_j * n_ben
    prec_j = tp / (tp + fp + 1e-12)
    f1_j   = 2 * prec_j * tpr_j / (prec_j + tpr_j + 1e-12)
    mcc_j  = float(
        (tp * (n_ben - fp) - fp * (n_mal - tp))
        / (np.sqrt((tp + fp) * (tp + (n_mal - tp)) * (n_ben - fp + fp) * ((n_mal - tp) + (n_ben - fp))) + 1e-12)
    )

    return {
        "auc":           auc_u,
        "auprc":         float("nan"),  # not computable from marginal ROC curves without sample alignment
        "f1_opt":        float(f1_j),
        "precision_opt": float(prec_j),
        "recall_opt":    tpr_j,
        "mcc_opt":       mcc_j,
        "tpr_at_fpr1":  _tpr_at(0.01),
        "tpr_at_fpr5":  _tpr_at(0.05),
    }


def pair_scores(s_a: np.ndarray, s_b: np.ndarray,
                rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    ia = rng.choice(len(s_a), n, replace=(len(s_a) < n))
    ib = rng.choice(len(s_b), n, replace=(len(s_b) < n))
    return s_a[ia], s_b[ib]


def fuse(a: np.ndarray, b: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "mean":          return (a + b) / 2
    if strategy == "weighted_mean": return W_MLRAN * a + W_RANSMAP * b
    if strategy == "max":           return np.maximum(a, b)
    if strategy == "min":           return np.minimum(a, b)
    if strategy == "product":       return a * b
    raise ValueError(strategy)


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Metrics at a fixed threshold derived from benign training distribution."""
    y_pred = (scores >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    mcc = float(matthews_corrcoef(y_true, y_pred))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return {
        "tpr": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "f1":  float(f1),
        "mcc": float(mcc),
    }


# --- aggregation --------------------------------------------------------------

METRIC_KEYS = ["auc", "auprc", "f1_opt", "precision_opt",
               "recall_opt", "mcc_opt", "tpr_at_fpr1", "tpr_at_fpr5",
               "deploy95_tpr", "deploy95_fpr", "deploy95_f1", "deploy95_mcc",
               "deploy99_tpr", "deploy99_fpr", "deploy99_f1", "deploy99_mcc"]


def aggregate(seed_runs: list[dict]) -> dict:
    out = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in seed_runs if k in r]
        if vals:
            out[f"mean_{k}"]    = round(float(np.mean(vals)),             4)
            # MAJ-05 fix: unbiased sample std (ddof=1) for the n=10 seed estimates.
            out[f"std_{k}"]     = round(float(np.std(vals, ddof=1)),      4)
            out[f"ci95_lo_{k}"] = round(float(np.percentile(vals,  2.5)), 4)
            out[f"ci95_hi_{k}"] = round(float(np.percentile(vals, 97.5)), 4)
    return out


# --- main ---------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- load data --
    print("Loading MLRan...")
    mlran = pd.read_csv(MLRAN_PATH, index_col=0)
    mfc   = [c for c in mlran.columns if c not in MLRAN_EXCLUDE]
    m_ben_raw = mlran[mlran["label"] == 0].reset_index(drop=True)
    # Group-based split: hash each row's feature vector → split by unique pattern group
    m_gids = np.array([
        hashlib.md5(row.astype(np.float32).tobytes()).hexdigest()
        for row in m_ben_raw[mfc].fillna(0).values
    ])
    m_ben = m_ben_raw.copy()
    m_ben["_gid"] = m_gids
    m_unique_gids = np.unique(m_gids)
    m_mal = {f: mlran[mlran["family"] == a].reset_index(drop=True)
             for f, a in MLRAN_FAMILY_MAP.items()}
    print(f"  MLRan: {len(m_ben)} benign ({len(m_unique_gids)} unique patterns) | feat={len(mfc)}")

    print("Loading RanSMAP...")
    ransmap = pd.read_csv(RANSMAP_PATH)
    num_cols = set(ransmap.select_dtypes(include="number").columns)
    rfc = [c for c in ransmap.columns if c in num_cols and c not in RANSMAP_EXCLUDE]
    r_ben = ransmap[ransmap["label"] == 0].reset_index(drop=True)
    r_mal = {f: ransmap[ransmap["family"] == a].reset_index(drop=True)
             for f, a in RANSMAP_FAMILY_MAP.items()}
    print(f"  RanSMAP: {len(r_ben)} benign | feat={len(rfc)}")

    # malware counts
    for f in CANONICAL_FAMILIES:
        print(f"  {f}: MLRan={len(m_mal[f])}  RanSMAP={len(r_mal[f])}")

    # storage: strategy → family → [per-seed metrics]
    STRATS = (["A_mlran", "A_ransmap", "A_union"]
              + [f"B_{s}" for s in FUSION_STRATEGIES]
              + ["C_concat"])
    runs: dict[str, dict[str, list]] = {s: {f: [] for f in CANONICAL_FAMILIES} for s in STRATS}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        # CRIT-02 fix: give each pairing purpose its own independent RNG stream
        # (derived from one SeedSequence) so Option B (score fusion) and Option C
        # (feature concat) no longer share/advance the same generator. This
        # removes the hidden cross-strategy coupling while staying fully seeded
        # and reproducible.
        rng_ctrain, rng_deploy, rng_b, rng_c = (
            np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
        )

        # MLRan: group-based split (split unique pattern groups, not rows)
        tr_gids, te_gids = train_test_split(
            m_unique_gids, test_size=0.2, random_state=seed
        )
        m_tr_mask = m_ben["_gid"].isin(set(tr_gids))
        Xmb_tr = m_ben[m_tr_mask][mfc].fillna(0.0).values
        Xmb_te = m_ben[~m_tr_mask][mfc].fillna(0.0).values

        # RanSMAP: stratified split by process type
        Xrb_tr_raw, Xrb_te_raw = train_test_split(
            r_ben[rfc].fillna(0.0).values, test_size=0.2,
            random_state=seed, stratify=r_ben["family"].values,
        )

        sc = StandardScaler().fit(Xrb_tr_raw)
        Xrb_tr, Xrb_te = sc.transform(Xrb_tr_raw), sc.transform(Xrb_te_raw)

        # Fit channel GMs with optimal K (from BIC/elbow analysis)
        gm_m = fit_gm(Xmb_tr, n_components=K_MLRAN,   covariance_type="diag", seed=seed)
        gm_r = fit_gm(Xrb_tr, n_components=K_RANSMAP, covariance_type="full", seed=seed)

        # Benign training scores (for MinMax normalization in Option B)
        sm_tr = score(gm_m, Xmb_tr)
        sr_tr = score(gm_r, Xrb_tr)

        # Option C: concat GM trained on bootstrap-paired benign
        n_cat = min(len(Xmb_tr), len(Xrb_tr))   # min(2040, 600) = 600
        im, ir = rng_ctrain.choice(len(Xmb_tr), n_cat, replace=False), rng_ctrain.choice(len(Xrb_tr), n_cat, replace=False)
        Xcat_tr = np.hstack([Xmb_tr[im], Xrb_tr[ir]])
        gm_cat  = fit_gm(Xcat_tr, n_components=1, covariance_type="diag", seed=seed)
        sc_cat  = score(gm_cat, Xcat_tr)   # benign train scores for normalization

        # Deployment-realistic thresholds: percentiles of fused benign TRAIN scores
        # Pair benign train scores (min-sized sample, fixed rng) then fuse per strategy
        n_tr_p = min(len(sm_tr), len(sr_tr))
        im_tr  = rng_deploy.choice(len(sm_tr), n_tr_p, replace=False)
        ir_tr  = rng_deploy.choice(len(sr_tr), n_tr_p, replace=False)
        sm_tr_n = minmax_norm(sm_tr, sm_tr[im_tr])
        sr_tr_n = minmax_norm(sr_tr, sr_tr[ir_tr])
        deploy_thresholds: dict[str, dict[int, float]] = {}
        for strat in FUSION_STRATEGIES:
            fused_ben_tr = fuse(sm_tr_n, sr_tr_n, strat)
            deploy_thresholds[strat] = {
                95: float(np.percentile(fused_ben_tr, 95)),
                99: float(np.percentile(fused_ben_tr, 99)),
            }

        for family in CANONICAL_FAMILIES:
            Xmmal = m_mal[family][mfc].fillna(0.0).values
            Xrmal = sc.transform(r_mal[family][rfc].fillna(0.0).values)

            n_mmal, n_rmal = len(Xmmal), len(Xrmal)
            n_mben_te, n_rben_te = len(Xmb_te), len(Xrb_te)

            # -- Option A: per-channel independent --
            Xmt = np.vstack([Xmb_te, Xmmal])
            ymt = np.array([0]*n_mben_te + [1]*n_mmal)
            m_a = compute_metrics(ymt, score(gm_m, Xmt), keep_curve=True)
            m_a["_n_ben"], m_a["_n_mal"] = n_mben_te, n_mmal

            Xrt = np.vstack([Xrb_te, Xrmal])
            yrt = np.array([0]*n_rben_te + [1]*n_rmal)
            m_b = compute_metrics(yrt, score(gm_r, Xrt), keep_curve=True)
            m_b["_n_ben"], m_b["_n_mal"] = n_rben_te, n_rmal

            m_u = union_metrics(m_a, m_b)

            runs["A_mlran"][family].append({k: m_a[k] for k in METRIC_KEYS if k in m_a})
            runs["A_ransmap"][family].append({k: m_b[k] for k in METRIC_KEYS if k in m_b})
            runs["A_union"][family].append(m_u)

            # -- Option B: normalized score fusion --
            sm_ben_norm = minmax_norm(sm_tr, score(gm_m, Xmb_te))
            sr_ben_norm = minmax_norm(sr_tr, score(gm_r, Xrb_te))
            sm_mal_norm = minmax_norm(sm_tr, score(gm_m, Xmmal))
            sr_mal_norm = minmax_norm(sr_tr, score(gm_r, Xrmal))

            n_mal_p = min(n_mmal, n_rmal)
            n_ben_p = min(n_mben_te, n_rben_te)

            pa_m, pb_m = pair_scores(sm_mal_norm, sr_mal_norm, rng_b, n_mal_p)
            pa_b, pb_b = pair_scores(sm_ben_norm, sr_ben_norm, rng_b, n_ben_p)

            for strat in FUSION_STRATEGIES:
                s_fused = np.concatenate([fuse(pa_b, pb_b, strat),
                                          fuse(pa_m, pb_m, strat)])
                y_fused = np.array([0]*n_ben_p + [1]*n_mal_p)
                m_fused = compute_metrics(y_fused, s_fused)
                # Add deployment-realistic threshold metrics
                for pct in (95, 99):
                    thr = deploy_thresholds[strat][pct]
                    dm = metrics_at_threshold(y_fused, s_fused, thr)
                    m_fused[f"deploy_thr{pct}"] = round(thr, 6)
                    for k, v in dm.items():
                        m_fused[f"deploy{pct}_{k}"] = round(float(v), 4)
                runs[f"B_{strat}"][family].append(m_fused)

            # -- Option C: feature concatenation --
            n_mal_c = min(n_mmal, n_rmal)
            n_ben_c = min(n_mben_te, n_rben_te)

            im_c, ir_c = pair_scores(np.arange(n_mmal), np.arange(n_rmal), rng_c, n_mal_c)
            ib_c, ibr_c = pair_scores(np.arange(n_mben_te), np.arange(n_rben_te), rng_c, n_ben_c)

            Xcmal = np.hstack([Xmmal[im_c.astype(int)], Xrmal[ir_c.astype(int)]])
            Xcben = np.hstack([Xmb_te[ib_c.astype(int)], Xrb_te[ibr_c.astype(int)]])
            Xc    = np.vstack([Xcben, Xcmal])
            yc    = np.array([0]*n_ben_c + [1]*n_mal_c)
            runs["C_concat"][family].append(compute_metrics(yc, score(gm_cat, Xc)))

            print(f"  {family:8s}  "
                  f"A_mlran={m_a['auc']:.4f}  A_ransmap={m_b['auc']:.4f}  "
                  f"A_union={m_u['auc']:.4f}  "
                  f"B_mean={runs['B_mean'][family][-1]['auc']:.4f}  "
                  f"C_concat={runs['C_concat'][family][-1]['auc']:.4f}")

    # -- latency benchmark (single-sample CPU, models from last seed) ----------
    N_TIMING = 1000
    x1m = Xmb_tr[0:1]   # 1 × D_mlran
    x1r = Xrb_tr[0:1]   # 1 × D_ransmap (already standardised)

    gm_m.score_samples(x1m)   # warm up
    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        gm_m.score_samples(x1m)
    lat_m_ms = (time.perf_counter() - t0) / N_TIMING * 1000

    gm_r.score_samples(x1r)   # warm up
    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        gm_r.score_samples(x1r)
    lat_r_ms = (time.perf_counter() - t0) / N_TIMING * 1000

    # B_product end-to-end: score_samples + MinMax norm + product fuse
    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        sm_n = minmax_norm(sm_tr, -gm_m.score_samples(x1m))
        sr_n = minmax_norm(sr_tr, -gm_r.score_samples(x1r))
        fuse(sm_n, sr_n, "product")
    lat_e2e_ms = (time.perf_counter() - t0) / N_TIMING * 1000

    print(f"\nLatency benchmark ({N_TIMING} calls, 1 sample, CPU):")
    print(f"  MLRan   K={K_MLRAN}  D={len(mfc)}:   {lat_m_ms:.4f} ms/sample")
    print(f"  RanSMAP K={K_RANSMAP}  D={len(rfc)}:  {lat_r_ms:.4f} ms/sample")
    print(f"  B_product end-to-end:         {lat_e2e_ms:.4f} ms/sample")

    latency_stats = {
        "n_timing_calls": N_TIMING,
        "note": "Single-sample CPU latency using models from final seed (seed 9). GM score computation is O(K x D).",
        "mlran_score_ms":   round(lat_m_ms,   4),
        "ransmap_score_ms": round(lat_r_ms,   4),
        "b_product_e2e_ms": round(lat_e2e_ms, 4),
        "mlran_k":   K_MLRAN,
        "ransmap_k": K_RANSMAP,
        "mlran_d":   len(mfc),
        "ransmap_d": len(rfc),
    }

    # -- aggregate and report --
    print("\n\n" + "="*90)
    print("AGGREGATED RESULTS (mean AUC ± std across 10 seeds)")
    print("="*90)

    results: dict = {}
    for strat in STRATS:
        strat_res: dict = {}
        for fam in CANONICAL_FAMILIES:
            strat_res[fam] = aggregate(runs[strat][fam])
        macro_aucs = [strat_res[f]["mean_auc"] for f in CANONICAL_FAMILIES]
        # Per-run AUC across all families × seeds for macro CI
        all_run_aucs = [r["auc"] for f in CANONICAL_FAMILIES for r in runs[strat][f] if "auc" in r]
        strat_res["__macro__"] = {
            "mean_auc":    round(float(np.mean(macro_aucs)),              4),
            "std_auc":     round(float(np.std(macro_aucs)),               4),
            "ci95_lo_auc": round(float(np.percentile(all_run_aucs,  2.5)), 4),
            "ci95_hi_auc": round(float(np.percentile(all_run_aucs, 97.5)), 4),
        }
        results[strat] = strat_res

    # Print AUC table
    hdr = f"  {'Strategy':<18}" + "".join(f"  {f:>9}" for f in CANONICAL_FAMILIES) + f"  {'Macro':>7}"
    print(hdr)
    print("  " + "-"*85)
    for strat in STRATS:
        row = f"  {strat:<18}"
        for f in CANONICAL_FAMILIES:
            v = results[strat][f]
            row += f"  {v['mean_auc']:>5.4f}±{v['std_auc']:.3f}"
        row += f"  {results[strat]['__macro__']['mean_auc']:>5.4f}"
        print(row)

    # Print full metrics for best strategy per family
    print("\n\nFULL METRICS - GaussianMixture per option/strategy")
    HDR2 = ["AUC", "AUPRC", "F1", "Recall", "Prec", "TPR@FPR1%", "TPR@FPR5%", "MCC"]
    KEYS2 = ["mean_auc", "mean_auprc", "mean_f1_opt", "mean_recall_opt",
             "mean_precision_opt", "mean_tpr_at_fpr1", "mean_tpr_at_fpr5", "mean_mcc_opt"]
    print(f"\n  {'Strategy':<18}" + "".join(f"  {h:>10}" for h in HDR2))
    print("  " + "-"*100)
    for strat in STRATS:
        macro = {}
        for k in KEYS2:
            vals = [results[strat][f].get(k, float("nan")) for f in CANONICAL_FAMILIES]
            macro[k] = float(np.nanmean(vals))
        row = f"  {strat:<18}" + "".join(f"  {macro[k]:>10.4f}" for k in KEYS2)
        print(row)

    # Paired Wilcoxon signed-rank test over the 50 fold x seed AUC runs
    # (5 families x 10 seeds): B_product against the next-best strategies.
    def _run_aucs(strat):
        return [r["auc"] for fam in CANONICAL_FAMILIES for r in runs[strat][fam] if "auc" in r]
    bp = _run_aucs("B_product")
    sig_tests = {}
    for other in ("A_union", "A_ransmap", "B_mean"):
        oth = _run_aucs(other)
        diffs = np.array(bp) - np.array(oth)
        n_nonzero = int(np.sum(np.abs(diffs) > 1e-12))
        entry = {"n": len(bp), "n_nonzero_diffs": n_nonzero,
                 "median_diff": round(float(np.median(diffs)), 6)}
        try:
            stat, p = wilcoxon(bp, oth, zero_method="wilcox")
            entry["wilcoxon_stat"] = round(float(stat), 4)
            entry["p_value"] = round(float(p), 4)
        except ValueError:
            entry["p_value"] = None   # all paired differences are zero (identical)
        sig_tests[f"B_product_vs_{other}"] = entry
    print("\nWilcoxon signed-rank (B_product vs next-best):")
    for k, v in sig_tests.items():
        print(f"  {k}: {v}")

    out_path = OUTPUT_DIR / "ano_combined_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "description": (
                "LOFO on 5 canonical families x 10 seeds. "
                f"GaussianMixture: MLRan K={K_MLRAN} diag (BIC/elbow), RanSMAP K={K_RANSMAP} full (BIC min + domain). "
                "MLRan: group-based split (unique feature vector hash, all 2,550 files kept). "
                "RanSMAP: stratified 80/20 split by process type. "
                "Option A: independent per-channel + probabilistic union. "
                "Option B: bootstrap score fusion (5 strategies, MinMax norm). "
                "Option C: feature concat (516-dim K=1 diag GaussianMixture). "
                "Metrics: AUC, AUPRC, F1, Precision, Recall, MCC, TPR@FPR1%, TPR@FPR5%."
            ),
            "n_seeds": len(SEEDS),
            "canonical_families": CANONICAL_FAMILIES,
            "fusion_strategies_B": FUSION_STRATEGIES,
            "gm_k_mlran":   K_MLRAN,
            "gm_k_ransmap": K_RANSMAP,
            "weights_B_weighted_mean": {"W_mlran": round(W_MLRAN, 4), "W_ransmap": round(W_RANSMAP, 4)},
            "results": results,
            "significance_tests": sig_tests,
            "latency": latency_stats,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
