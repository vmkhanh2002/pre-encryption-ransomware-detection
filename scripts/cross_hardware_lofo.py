"""Leave-one-CPU-out generalization check for the RanSMAP channel.

The main RanSMAP evaluation pools all hardware configurations. This script
asks whether the learned benign I/O profile transfers to unseen hardware:
it trains the RanSMAP GMM on benign traces from three of the four CPU
families and tests on the held-out CPU's benign traces plus all malware
(pooled across CPUs). It reports AUC and the false-positive rate of a
Deploy-95 threshold calibrated on the training CPUs (paper, Section 5.2).
"""
import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import K_RANSMAP, N_INIT, REG_COVAR, RESULTS_DIR, TEST_SIZE  # noqa: E402
from utils import load_ransmap_clean  # noqa: E402

OUT = RESULTS_DIR / "cross_hardware_lofo.json"

df, feat_cols = load_ransmap_clean(drop_zero_var=True)
X_raw = df[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
is_benign = df["label"].to_numpy() == 0
cpu = df["cpu"].to_numpy()
fam = df["family"].to_numpy()

CPUS = sorted(df["cpu"].unique())
SPLIT_SEED = 42  # one fixed split and fit per held-out CPU
K = K_RANSMAP


def nll(gm, Xq):
    return -gm.score_samples(Xq)


results = {}
for held_out_cpu in CPUS:
    tr_mask = is_benign & (cpu != held_out_cpu)
    te_ben_mask = is_benign & (cpu == held_out_cpu)
    mal_mask = ~is_benign

    # fit StandardScaler strictly on the training benign split
    scaler = StandardScaler().fit(X_raw[tr_mask])
    Xtr = scaler.transform(X_raw[tr_mask])
    gm = GaussianMixture(
        K, covariance_type="full", reg_covar=REG_COVAR, n_init=N_INIT, random_state=SPLIT_SEED
    ).fit(Xtr)

    Xte = scaler.transform(np.vstack([X_raw[te_ben_mask], X_raw[mal_mask]]))
    yte = np.concatenate([np.zeros(te_ben_mask.sum()), np.ones(mal_mask.sum())])
    s = nll(gm, Xte)
    auc = roc_auc_score(yte, s)

    s_tr = nll(gm, Xtr)
    tau95 = float(np.percentile(s_tr, 95))
    fpr_at_tau95 = float(np.mean(s[: te_ben_mask.sum()] >= tau95))
    tpr_at_tau95 = float(np.mean(s[te_ben_mask.sum() :] >= tau95))

    results[held_out_cpu] = {
        "n_train_benign": int(tr_mask.sum()),
        "n_test_benign_heldout_cpu": int(te_ben_mask.sum()),
        "n_malware": int(mal_mask.sum()),
        "auc": float(auc),
        "deploy95_fpr_on_heldout_cpu_benign": fpr_at_tau95,
        "deploy95_tpr": tpr_at_tau95,
    }
    print(
        f"held-out CPU={held_out_cpu}: train_n={tr_mask.sum()} test_ben_n={te_ben_mask.sum()} "
        f"AUC={auc:.4f} deploy95_FPR={fpr_at_tau95:.4f} deploy95_TPR={tpr_at_tau95:.4f}",
        flush=True,
    )

# Baseline: pooled (stratified by family, scaled on train)
tr_idx, te_idx = train_test_split(
    np.where(is_benign)[0],
    test_size=TEST_SIZE,
    random_state=SPLIT_SEED,
    stratify=fam[is_benign],
)
scaler_p = StandardScaler().fit(X_raw[tr_idx])
gm_pooled = GaussianMixture(
    K, covariance_type="full", reg_covar=REG_COVAR, n_init=N_INIT, random_state=SPLIT_SEED
).fit(scaler_p.transform(X_raw[tr_idx]))
Xte_pooled = scaler_p.transform(np.vstack([X_raw[te_idx], X_raw[~is_benign]]))
yte_pooled = np.concatenate([np.zeros(len(te_idx)), np.ones((~is_benign).sum())])
auc_pooled = roc_auc_score(yte_pooled, nll(gm_pooled, Xte_pooled))
print(f"\npooled (stratified, scaled baseline): AUC={auc_pooled:.4f}", flush=True)

out = {
    "description": __doc__.split("\n")[0],
    "per_held_out_cpu": results,
    "macro_auc_cross_hardware": float(np.mean([v["auc"] for v in results.values()])),
    "macro_deploy95_fpr_cross_hardware": float(
        np.mean([v["deploy95_fpr_on_heldout_cpu_benign"] for v in results.values()])
    ),
    "pooled_stratified_baseline_auc": float(auc_pooled),
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items() if k != "per_held_out_cpu"}, indent=2))
print("saved ->", OUT)
