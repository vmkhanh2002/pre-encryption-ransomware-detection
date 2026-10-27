"""Randomized control for whitebox_feature_freeze.py.

Freezing the 5 write-throughput features collapses nearest-mode evasion.
That only indicates those features matter specifically if freezing other
5-feature subsets does not cause a similar collapse. This script draws
N_DRAWS random 5-feature subsets per seed (excluding the write-throughput
features), reruns the identical masked bisection attack for each, and
reports where the write-frozen result (read from
whitebox_feature_freeze.json) falls in that distribution.
"""
import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import CANONICAL_FAMILIES, N_INIT, REG_COVAR, RESULTS_DIR, SEEDS, TEST_SIZE  # noqa: E402
from utils import load_ransmap_clean, mahal_sq_to_centers  # noqa: E402

OUT = RESULTS_DIR / "whitebox_feature_freeze_control.json"
ABLATION_JSON = RESULTS_DIR / "whitebox_feature_freeze.json"
N_DRAWS = 20
CANONICAL = CANONICAL_FAMILIES

df, feat_cols = load_ransmap_clean(drop_zero_var=True)
X_raw = df[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
fam = df["family"].to_numpy()
is_benign = df["label"].to_numpy() == 0

WRITE_SIG_FEATS = [
    "w_bytes_per_s",
    "w_total_bytes",
    "w_io_count",
    "w_io_per_s",
    "w_sequential_frac",
]
write_sig_idx = set(feat_cols.index(f) for f in WRITE_SIG_FEATS)
non_write_sig_idx = [i for i in range(len(feat_cols)) if i not in write_sig_idx]


def nll(gm, Xq):
    return -gm.score_samples(Xq)


def solve_alpha_masked(gm, X0, T, tau, mask, max_a=1.0):
    def f(a):
        aa = np.asarray(a)
        if aa.ndim == 0:
            return nll(gm, X0 + float(aa) * mask[None, :] * (T - X0))
        return nll(gm, X0 + aa[:, None] * mask[None, :] * (T - X0))

    n = len(X0)
    alphas = np.zeros(n)
    inside = f(0.0) <= tau
    reachable = f(max_a) <= tau
    todo = (~inside) & reachable
    lo = np.zeros(n)
    hi = np.full(n, max_a)
    for _ in range(30):
        mid = (lo + hi) / 2
        ok = f(mid) <= tau
        hi = np.where(ok & todo, mid, hi)
        lo = np.where(~ok & todo, mid, lo)
    alphas[todo] = hi[todo]
    success = inside | reachable
    return alphas, success


res = {
    "description": __doc__.split("\n")[0],
    "n_draws_per_seed": N_DRAWS,
    "write_sig_features": WRITE_SIG_FEATS,
    "per_seed": {},
}
random_success_all = []

for seed in SEEDS:
    idx_b = np.where(is_benign)[0]
    # stratify + StandardScaler on benign train
    tr_b, _ = train_test_split(
        idx_b, test_size=TEST_SIZE, random_state=seed, stratify=fam[idx_b]
    )
    scaler = StandardScaler().fit(X_raw[tr_b])
    X = scaler.transform(X_raw)

    gm = GaussianMixture(
        5, covariance_type="full", reg_covar=REG_COVAR, n_init=N_INIT, random_state=seed
    ).fit(X[tr_b])
    mus = gm.means_
    s_tr = nll(gm, X[tr_b])
    tau = float(np.median(s_tr))

    mal_idx = np.array([i for i in np.where(~is_benign)[0] if fam[i] in CANONICAL])
    Xmal = X[mal_idx]

    # Mahalanobis distance using gm.precisions_
    kstar = mahal_sq_to_centers(Xmal, gm).argmin(axis=1)
    T_mode = mus[kstar]

    rng = np.random.default_rng(seed)
    draw_success = []
    for d in range(N_DRAWS):
        frozen5 = rng.choice(non_write_sig_idx, size=5, replace=False)
        mask = np.ones(len(feat_cols))
        mask[frozen5] = 0.0
        _, succ = solve_alpha_masked(gm, Xmal, T_mode, tau, mask)
        draw_success.append(float(succ.mean()))
    random_success_all.extend(draw_success)
    res["per_seed"][f"seed{seed}"] = {
        "random_freeze_success_rates": draw_success,
        "random_freeze_mean": float(np.mean(draw_success)),
        "random_freeze_min": float(np.min(draw_success)),
    }
    print(
        f"seed {seed}: random-freeze success mean={np.mean(draw_success):.3f} "
        f"min={np.min(draw_success):.3f} max={np.max(draw_success):.3f}",
        flush=True,
    )

random_success_all = np.array(random_success_all)
if ABLATION_JSON.exists():
    write_sig_success_mean = float(
        json.load(open(ABLATION_JSON))["summary"]["write_sig_frozen_success_mean"]
    )
else:
    write_sig_success_mean = 0.0

res["summary"] = {
    "n_total_random_draws": len(random_success_all),
    "random_freeze_success_mean": float(random_success_all.mean()),
    "random_freeze_success_std": float(random_success_all.std(ddof=1)),
    "random_freeze_success_min": float(random_success_all.min()),
    "random_freeze_success_max": float(random_success_all.max()),
    "frac_random_draws_below_write_sig_point": float(
        np.mean(random_success_all <= write_sig_success_mean)
    ),
    "write_sig_frozen_success_mean_reference": write_sig_success_mean,
}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res["summary"], indent=2))
print("saved ->", OUT)
