"""Zero-knowledge throttling (rate-limiting) evasion sweep.

The threat model lets the attacker encrypt intermittently (paper, Section
5.1). Unlike the Level-2/3 attacks, throttling needs no knowledge of the
detector, the GMM, or any benign statistic: the same encryption I/O is
spread over more wall-clock time (e.g. sleeping between chunks).

Physical model: total work is fixed (the same files must be encrypted, so
total bytes, I/O count, size distribution and sector footprint cannot
shrink), but duration can grow by a slowdown factor k. This divides the rate
features (io_per_s, bytes_per_s) by k and multiplies duration_s by k, for
reads and writes; all other features keep the malware's own values. It
corresponds to the post-encryption delay knob of Hirano and Kobayashi
(IEEE CSR 2025) in feature space (paper, Section 4.2).
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
from config import CANONICAL_FAMILIES, N_INIT, REG_COVAR, RESULTS_DIR, SEEDS, TEST_SIZE  # noqa: E402
from utils import load_ransmap_clean  # noqa: E402

OUT = RESULTS_DIR / "throttling_evasion.json"
CANONICAL = CANONICAL_FAMILIES
KS = [1, 2, 5, 10, 20, 50, 100]

df, feat_cols = load_ransmap_clean(drop_zero_var=True)
X_raw = df[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
fam = df["family"].to_numpy()
is_benign = df["label"].to_numpy() == 0

idx_r_io_per_s = feat_cols.index("r_io_per_s")
idx_w_io_per_s = feat_cols.index("w_io_per_s")
idx_r_bytes_per_s = feat_cols.index("r_bytes_per_s")
idx_w_bytes_per_s = feat_cols.index("w_bytes_per_s")
idx_r_duration_s = feat_cols.index("r_duration_s")
idx_w_duration_s = feat_cols.index("w_duration_s")


def throttle_physical(Xmal_phys, k):
    """Slow down by factor k in physical feature space: duration *= k, rate features /= k."""
    Xt = Xmal_phys.copy()
    Xt[:, idx_r_duration_s] *= k
    Xt[:, idx_w_duration_s] *= k
    Xt[:, idx_r_io_per_s] /= k
    Xt[:, idx_w_io_per_s] /= k
    Xt[:, idx_r_bytes_per_s] /= k
    Xt[:, idx_w_bytes_per_s] /= k
    return Xt


def nll(gm, Xq):
    return -gm.score_samples(Xq)


res = {
    "description": __doc__.split("\n")[0],
    "ks": KS,
    "per_seed": {},
}
agg = {k: {"auc": [], "evasion_frac": [], "detected_frac_at_tau": []} for k in KS}

for seed in SEEDS:
    idx_b = np.where(is_benign)[0]
    # stratify + StandardScaler fit on benign train split
    tr_b, te_b = train_test_split(
        idx_b, test_size=TEST_SIZE, random_state=seed, stratify=fam[idx_b]
    )
    scaler = StandardScaler().fit(X_raw[tr_b])
    Xtr = scaler.transform(X_raw[tr_b])
    Xte_b = scaler.transform(X_raw[te_b])

    gm = GaussianMixture(
        5, covariance_type="full", reg_covar=REG_COVAR, n_init=N_INIT, random_state=seed
    ).fit(Xtr)
    s_tr = nll(gm, Xtr)
    tau = float(np.median(s_tr))

    mal_idx = np.array([i for i in np.where(~is_benign)[0] if fam[i] in CANONICAL])
    Xmal_raw = X_raw[mal_idx]
    yte = np.concatenate([np.zeros(len(te_b)), np.ones(len(mal_idx))])
    s_te_b = nll(gm, Xte_b)

    seed_out = {}
    for k in KS:
        Xadv_raw = throttle_physical(Xmal_raw, k) if k > 1 else Xmal_raw
        Xadv = scaler.transform(Xadv_raw)
        s_adv = nll(gm, Xadv)
        auc = roc_auc_score(yte, np.concatenate([s_te_b, s_adv]))
        evasion_frac = float(np.mean(s_adv < tau))
        detected_frac = float(np.mean(s_adv >= tau))
        seed_out[f"k{k}"] = {
            "auc": float(auc),
            "evasion_frac": evasion_frac,
            "detected_frac_at_tau": detected_frac,
        }
        agg[k]["auc"].append(auc)
        agg[k]["evasion_frac"].append(evasion_frac)
        agg[k]["detected_frac_at_tau"].append(detected_frac)
    res["per_seed"][f"seed{seed}"] = seed_out
    print(
        f"seed {seed}: "
        + " | ".join(
            f"k={k} auc={seed_out[f'k{k}']['auc']:.3f} evade={seed_out[f'k{k}']['evasion_frac']:.2f}"
            for k in KS
        ),
        flush=True,
    )

res["summary"] = {
    f"k{k}": {
        "auc_mean": float(np.mean(agg[k]["auc"])),
        "evasion_frac_mean": float(np.mean(agg[k]["evasion_frac"])),
        "detected_frac_at_tau_mean": float(np.mean(agg[k]["detected_frac_at_tau"])),
    }
    for k in KS
}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res["summary"], indent=2))
print("saved ->", OUT)
