"""
Multi-model anomaly detection evaluation for MLRan and RanSMAP channels.

5 anomaly models x 10 seeds per channel.

Evaluation design:
  MLRan  - group-based split: split by unique feature vector group, keeping all
            2,550 files. No feature-vector pattern appears in both train and test.
  RanSMAP - stratified 80/20 split by process type (Firefox/Idle/Office/SDelete/Zip),
            ensuring balanced representation across all seeds.

Note: 5-family LOFO (Conti/Ryuk/LockBit/REvil/WannaCry) is reserved
      for the combined model only - these families are the intersection of both channels.
"""

from config import MLRAN_CSV, RANSMAP_CSV, RESULTS_DIR, SEEDS, TEST_SIZE, K_MLRAN, K_RANSMAP  # noqa: E402
import hashlib
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler

# MLRan:   BIC descends through K=12 (no minimum); elbow at K=3 (delta K=3->4 = 16% of K=1->2 jump)
#          diag covariance: full would need 116k params/component for 482-dim binary features
# RanSMAP: BIC minimum at K=5; domain knowledge = 5 process types (Firefox/Idle/Office/SDelete/Zip)

MLRAN_PATH   = MLRAN_CSV
RANSMAP_PATH = RANSMAP_CSV
OUTPUT_DIR   = RESULTS_DIR

MLRAN_EXCLUDE   = {"label", "family", "sample_id", "sha256", "split", "STRING:labels", "dataset"}
RANSMAP_EXCLUDE = {"label", "family", "trace_path", "dataset"}

# -- Evaluation constants -------------------------------------------------------
DEPLOY_PERCENTILES = (95, 99)

# -- Model hyperparameters ------------------------------------------------------
IF_N_ESTIMATORS     = 100
OC_SVM_NU           = 0.1   # shared by OneClassSVM and SGDOneClassSVM
LOF_N_NEIGHBORS     = 20
SGD_MAX_ITER        = 1000
GM_REG_COVAR_MULTI  = 1e-3  # K > 1: guard against near-singular covariance
GM_REG_COVAR_SINGLE = 1e-6  # K = 1: tighter regularization
GM_N_INIT_MULTI     = 5     # n_init for K > 1 to escape EM local optima

# -- Aggregation metric keys ----------------------------------------------------
METRIC_KEYS = [
    "auc", "auprc", "f1_opt", "precision_opt", "recall_opt",
    "mcc_opt", "tpr_at_fpr1", "tpr_at_fpr5",
    *[f"deploy{p}_{k}" for p in DEPLOY_PERCENTILES for k in ("tpr", "fpr", "f1", "mcc")],
]


def build_models(seed: int, gm_k: int = 1, gm_cov: str = "full") -> dict:
    gm_reg   = GM_REG_COVAR_MULTI  if gm_k > 1 else GM_REG_COVAR_SINGLE
    gm_ninit = GM_N_INIT_MULTI     if gm_k > 1 else 1
    return {
        "IsolationForest": IsolationForest(
            n_estimators=IF_N_ESTIMATORS, contamination="auto", random_state=seed
        ),
        "OneClassSVM": OneClassSVM(
            kernel="rbf", nu=OC_SVM_NU, gamma="scale"
        ),
        "LOF": LocalOutlierFactor(
            n_neighbors=LOF_N_NEIGHBORS, novelty=True, contamination="auto"
        ),
        "GaussianMixture": GaussianMixture(
            n_components=gm_k,
            covariance_type=gm_cov,
            reg_covar=gm_reg,
            n_init=gm_ninit,
            random_state=seed,
        ),
        "SGDOneClassSVM": SGDOneClassSVM(
            nu=OC_SVM_NU, random_state=seed, max_iter=SGD_MAX_ITER
        ),
    }


def anomaly_score(model, X: np.ndarray) -> np.ndarray:
    """Unified anomaly score: higher = more anomalous."""
    name = type(model).__name__
    if name == "GaussianMixture":
        return -model.score_samples(X)
    if hasattr(model, "score_samples"):
        return -model.score_samples(X)
    if hasattr(model, "decision_function"):
        return -model.decision_function(X)
    raise ValueError(f"No scoring method for {name}")


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Metrics at a fixed threshold (deployment-realistic - no test-set knowledge)."""
    y_pred = (scores >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    mcc = float(matthews_corrcoef(y_true, y_pred))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    denom_tpr = tp + fn
    denom_fpr = fp + tn
    return {
        "tpr":       float(tp / denom_tpr) if denom_tpr else 0.0,
        "fpr":       float(fp / denom_fpr) if denom_fpr else 0.0,
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "mcc":       float(mcc),
    }


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc   = float(roc_auc_score(y_true, scores))
    auprc = float(average_precision_score(y_true, scores))

    j_idx   = int(np.argmax(tpr - fpr))
    thr_opt = float(thresholds[j_idx])
    y_pred  = (scores >= thr_opt).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    mcc = float(matthews_corrcoef(y_true, y_pred))

    def tpr_at_fpr(target: float) -> float:
        return float(np.interp(target, fpr, tpr))

    return {
        "auc":           auc,
        "auprc":         auprc,
        "f1_opt":        float(f1),
        "precision_opt": float(prec),
        "recall_opt":    float(rec),
        "mcc_opt":       mcc,
        "tpr_at_fpr1":   tpr_at_fpr(0.01),
        "tpr_at_fpr5":   tpr_at_fpr(0.05),
        "threshold_opt": thr_opt,
    }


def _vec_group_ids(df: pd.DataFrame, feat_cols: list) -> np.ndarray:
    """MD5 hash per row → group label for each unique feature vector."""
    return np.array([
        hashlib.md5(row.astype(np.float32).tobytes()).hexdigest()
        for row in df[feat_cols].fillna(0).values
    ])


def eval_channel(
    df: pd.DataFrame,
    feat_cols: list,
    channel_name: str,
    scale_features: bool = False,
    split_mode: str = "group",   # "group" | "stratified"
    gm_k: int = 1,
    gm_cov: str = "full",
) -> dict:
    benign  = df[df["label"] == 0].reset_index(drop=True)
    malware = df[df["label"] == 1].reset_index(drop=True)
    X_mal   = malware[feat_cols].fillna(0.0).values

    if split_mode == "group":
        # Pre-compute group IDs once; split is over unique patterns (not rows)
        gids        = _vec_group_ids(benign, feat_cols)
        unique_gids = np.unique(gids)
        benign      = benign.copy()
        benign["_gid"] = gids
        print(f"\n  Benign: {len(benign)} rows ({len(unique_gids)} unique patterns) | "
              f"Malware: {len(malware)} ({malware['family'].nunique()} families) | "
              f"Features: {len(feat_cols)}")
    else:
        print(f"\n  Benign: {len(benign)} | Malware: {len(malware)} "
              f"({malware['family'].nunique()} families) | Features: {len(feat_cols)}")

    results: dict = {}

    for model_name in build_models(0, gm_k=gm_k, gm_cov=gm_cov).keys():
        seed_metrics: list[dict] = []
        t0 = time.time()

        for seed in SEEDS:
            if split_mode == "group":
                tr_gids, te_gids = train_test_split(
                    unique_gids, test_size=TEST_SIZE, random_state=seed
                )
                mask        = benign["_gid"].isin(set(tr_gids))
                X_ben_train = benign[mask][feat_cols].fillna(0.0).values
                X_ben_test  = benign[~mask][feat_cols].fillna(0.0).values
            else:  # stratified
                X_ben_train, X_ben_test = train_test_split(
                    benign[feat_cols].fillna(0.0).values,
                    test_size=TEST_SIZE,
                    random_state=seed,
                    stratify=benign["family"].values,
                )

            if scale_features:
                scaler      = StandardScaler()
                X_ben_train = scaler.fit_transform(X_ben_train)
                X_ben_test  = scaler.transform(X_ben_test)
                X_mal_sc    = scaler.transform(X_mal)
            else:
                X_mal_sc = X_mal

            X_test = np.vstack([X_ben_test, X_mal_sc])
            y_test = np.array([0] * len(X_ben_test) + [1] * len(X_mal_sc))

            model = build_models(seed, gm_k=gm_k, gm_cov=gm_cov)[model_name]
            try:
                model.fit(X_ben_train)
                train_scores = anomaly_score(model, X_ben_train)
                scores = anomaly_score(model, X_test)
                m = compute_metrics(y_test, scores)
                # Deployment-realistic thresholds: set from benign train distribution
                # 95th pct → ~5% FPR on training benign; 99th → ~1% FPR
                for pct in DEPLOY_PERCENTILES:
                    thr = float(np.percentile(train_scores, pct))
                    dm = metrics_at_threshold(y_test, scores, thr)
                    m[f"deploy_thr{pct}"] = round(thr, 6)
                    for k, v in dm.items():
                        m[f"deploy{pct}_{k}"] = round(float(v), 4)
                seed_metrics.append(m)
            except Exception as e:
                print(f"    [{model_name}] seed={seed} FAILED: {e}")

        elapsed = time.time() - t0
        if not seed_metrics:
            results[model_name] = {"error": "all seeds failed", "elapsed_sec": round(elapsed, 1)}
            continue

        agg: dict = {"elapsed_sec": round(elapsed, 1), "n_seeds": len(seed_metrics)}
        for key in METRIC_KEYS:
            vals = [m[key] for m in seed_metrics]
            agg[f"mean_{key}"]     = round(float(np.mean(vals)),             4)
            agg[f"std_{key}"]      = round(float(np.std(vals, ddof=1)),      4)  # MAJ-05: unbiased sample std (n=10 seeds)
            agg[f"ci95_lo_{key}"]  = round(float(np.percentile(vals,  2.5)), 4)
            agg[f"ci95_hi_{key}"]  = round(float(np.percentile(vals, 97.5)), 4)
        agg["per_seed"] = seed_metrics
        results[model_name] = agg

        print(
            f"  [{channel_name}] {model_name:20s}: "
            f"AUC={agg['mean_auc']:.4f}+-{agg['std_auc']:.4f}  "
            f"AUPRC={agg['mean_auprc']:.4f}  "
            f"F1={agg['mean_f1_opt']:.4f}  "
            f"Rec={agg['mean_recall_opt']:.4f}  "
            f"TPR@FPR1%={agg['mean_tpr_at_fpr1']:.4f}  "
            f"MCC={agg['mean_mcc_opt']:.4f}  ({elapsed:.1f}s)"
        )

    return results


def eval_channel_per_family(
    df: pd.DataFrame,
    feat_cols: list,
    channel_name: str,
    scale_features: bool = False,
    split_mode: str = "group",
    gm_k: int = 1,
    gm_cov: str = "full",
) -> list:
    """Per-family AUC sweep using GaussianMixture (optimal K) across all families.

    For each malware family:
      train = benign training split (group-based or stratified, same as main eval)
      test  = benign test split + all samples of this family
      → AUC averaged over 10 seeds

    Families with < 2 samples are skipped (AUC undefined).
    """
    benign   = df[df["label"] == 0].reset_index(drop=True)
    malware  = df[df["label"] == 1].reset_index(drop=True)
    families = sorted(malware["family"].unique())

    if split_mode == "group":
        gids        = _vec_group_ids(benign, feat_cols)
        unique_gids = np.unique(gids)
        benign      = benign.copy()
        benign["_gid"] = gids

    print(f"\n  [{channel_name}] Per-family AUC sweep - "
          f"{len(families)} families, {len(SEEDS)} seeds, "
          f"GaussianMixture(K={gm_k}, cov={gm_cov})")

    records = []

    for fam in families:
        X_fam = malware[malware["family"] == fam][feat_cols].fillna(0.0).values
        n_fam = len(X_fam)
        if n_fam < 2:
            print(f"    [{fam:25s}] SKIPPED (n={n_fam})")
            continue

        seed_aucs = []
        for seed in SEEDS:
            if split_mode == "group":
                tr_gids, _ = train_test_split(unique_gids, test_size=TEST_SIZE, random_state=seed)
                mask        = benign["_gid"].isin(set(tr_gids))
                X_ben_train = benign[mask][feat_cols].fillna(0.0).values
                X_ben_test  = benign[~mask][feat_cols].fillna(0.0).values
            else:
                X_ben_train, X_ben_test = train_test_split(
                    benign[feat_cols].fillna(0.0).values,
                    test_size=TEST_SIZE, random_state=seed,
                    stratify=benign["family"].values,
                )

            if scale_features:
                scaler      = StandardScaler()
                X_ben_train = scaler.fit_transform(X_ben_train)
                X_ben_test  = scaler.transform(X_ben_test)
                X_fam_sc    = scaler.transform(X_fam)
            else:
                X_fam_sc = X_fam

            X_test = np.vstack([X_ben_test, X_fam_sc])
            y_test = np.array([0] * len(X_ben_test) + [1] * n_fam)

            gm = build_models(seed, gm_k=gm_k, gm_cov=gm_cov)["GaussianMixture"]
            try:
                gm.fit(X_ben_train)
                scores = -gm.score_samples(X_test)
                seed_aucs.append(float(roc_auc_score(y_test, scores)))
            except Exception:
                pass

        if not seed_aucs:
            continue

        rec = {
            "channel":  channel_name,
            "family":   fam,
            "n_malware": n_fam,
            "mean_auc": round(float(np.mean(seed_aucs)), 4),
            "std_auc":  round(float(np.std(seed_aucs, ddof=1)),  4),  # MAJ-05
            "ci95_lo":  round(float(np.percentile(seed_aucs, 2.5)),  4),
            "ci95_hi":  round(float(np.percentile(seed_aucs, 97.5)), 4),
            "min_auc":  round(float(np.min(seed_aucs)), 4),
            "max_auc":  round(float(np.max(seed_aucs)), 4),
            "n_seeds":  len(seed_aucs),
        }
        records.append(rec)
        print(
            f"    [{fam:25s}]  n={n_fam:4d}  "
            f"AUC={rec['mean_auc']:.4f}±{rec['std_auc']:.4f}  "
            f"95%CI=[{rec['ci95_lo']:.4f},{rec['ci95_hi']:.4f}]"
        )

    macro = float(np.mean([r["mean_auc"] for r in records])) if records else float("nan")
    print(f"\n  [{channel_name}] Macro AUC across {len(records)} families: {macro:.4f}")
    return records



def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- MLRan ---
    print("Loading MLRan...")
    mlran = pd.read_csv(MLRAN_PATH, index_col=0)
    mlran_feat_cols = [c for c in mlran.columns if c not in MLRAN_EXCLUDE]
    print(f"  Shape: {mlran.shape}, feature cols: {len(mlran_feat_cols)}")

    print(f"\n=== MLRan channel - group-based split (unique feature vector) | GaussianMixture(K={K_MLRAN}, diag) ===")
    mlran_results = eval_channel(
        mlran, mlran_feat_cols, "ano_mlran",
        scale_features=False, split_mode="group",
        gm_k=K_MLRAN, gm_cov="diag",
    )

    # --- RanSMAP ---
    print("\nLoading RanSMAP...")
    ransmap = pd.read_csv(RANSMAP_PATH)
    numeric_cols = set(ransmap.select_dtypes(include="number").columns)
    ransmap_feat_cols = [c for c in ransmap.columns
                         if c in numeric_cols and c not in RANSMAP_EXCLUDE]
    print(f"  Shape: {ransmap.shape}, feature cols: {len(ransmap_feat_cols)}")

    print(f"\n=== RanSMAP channel - stratified split by process type (5 models x 10 seeds) | GaussianMixture(K={K_RANSMAP}, full) ===")
    ransmap_results = eval_channel(
        ransmap, ransmap_feat_cols, "ano_ransmap",
        scale_features=True, split_mode="stratified",
        gm_k=K_RANSMAP, gm_cov="full",
    )

    # --- Per-family AUC breakdown (GaussianMixture, optimal K) ---
    print(f"\n=== MLRan - Per-family AUC ({len(mlran['family'].dropna().unique())} families) ===")
    mlran_per_family = eval_channel_per_family(
        mlran, mlran_feat_cols, "ano_mlran",
        scale_features=False, split_mode="group",
        gm_k=K_MLRAN, gm_cov="diag",
    )

    print(f"\n=== RanSMAP - Per-family AUC ({len(ransmap[ransmap['label']==1]['family'].dropna().unique())} families) ===")
    ransmap_per_family = eval_channel_per_family(
        ransmap, ransmap_feat_cols, "ano_ransmap",
        scale_features=True, split_mode="stratified",
        gm_k=K_RANSMAP, gm_cov="full",
    )

    per_family_path = OUTPUT_DIR / "per_family_auc.csv"
    per_family_df   = pd.DataFrame(mlran_per_family + ransmap_per_family)
    per_family_df.to_csv(per_family_path, index=False)
    print(f"\nSaved -> {per_family_path}")

    # --- Summary tables ---
    SHOW = ["mean_auc", "mean_auprc", "mean_f1_opt", "mean_recall_opt",
            "mean_tpr_at_fpr1", "mean_tpr_at_fpr5", "mean_mcc_opt"]
    HDR  = ["AUC", "AUPRC", "F1", "Recall", "TPR@FPR1%", "TPR@FPR5%", "MCC"]

    for ch_name, ch_res in [("MLRan", mlran_results), ("RanSMAP", ransmap_results)]:
        print(f"\n=== SUMMARY: {ch_name} ===")
        header = f"  {'Model':<20}" + "".join(f"  {h:>10}" for h in HDR)
        print(header)
        print("  " + "-" * (20 + 12 * len(HDR)))
        for m, v in ch_res.items():
            if "error" in v:
                print(f"  {m:<20}  FAILED")
                continue
            row = f"  {m:<20}" + "".join(f"  {v.get(k, float('nan')):>10.4f}" for k in SHOW)
            print(row)

    output = {
        "description": (
            "5 anomaly models x 10 seeds per channel. "
            f"GaussianMixture: MLRan K={K_MLRAN} diag (BIC/elbow), RanSMAP K={K_RANSMAP} full (BIC min + domain). "
            "MLRan: group-based split by unique feature vector - all 2,550 files kept, "
            "no feature-vector pattern leaks from train to test. "
            "RanSMAP: stratified 80/20 split by process type (Firefox/Idle/Office/SDelete/Zip), "
            "all five types present in both train and test for every seed. "
            "Specificity reported at oracle (Youden J) and deploy-95 thresholds."
        ),
        "gm_k_mlran":   K_MLRAN,
        "gm_k_ransmap": K_RANSMAP,
        "n_seeds":        len(SEEDS),
        "seeds":          SEEDS,
        "split_mlran":    "group-based (unique feature vector hash)",
        "split_ransmap":  "stratified by process type",
        "ano_mlran":      mlran_results,
        "ano_ransmap":    ransmap_results,
        "per_family_auc_mlran":   mlran_per_family,
        "per_family_auc_ransmap": ransmap_per_family,
    }

    out_path = OUTPUT_DIR / "ano_models_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
