"""Fused AUC with a near-duplicate-hardened MLRan split.

Compares the fused AUC under the exact-hash group split used in the paper
with an r=1 Hamming-hardened grouping (vectors within one bit merged into
the same group), with the same protocol (5-family LOFO, 10 seeds,
benign-only training) and the learned meta-GMM fusion of learned_fusion.py.
Backs the paper's statement that near-duplicate hardening leaves the fused
result unchanged (Section 5.2).
"""
import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import CANONICAL_FAMILIES, K_MLRAN, K_RANSMAP, META_REG_COVAR, N_INIT, REG_COVAR, RESULTS_DIR, SEEDS, TEST_SIZE, selected_meta_k  # noqa: E402
from utils import load_mlran, load_ransmap_clean, minmax_norm, oof_benign_norm_scores_ransmap  # noqa: E402

OUT = RESULTS_DIR / "hardened_fusion.json"

CANON_R = CANONICAL_FAMILIES
META_K = selected_meta_k()

dfm, fm = load_mlran()
Xm = dfm[fm].fillna(0).to_numpy(np.uint8)
ym = dfm["label"].to_numpy()
fam_m = np.array([str(f).lower() for f in dfm["family"].to_numpy()])

# build hardened groups at r=1 over unique BENIGN vectors only
Xm_ben = Xm[ym == 0]
uniq_b, inv_b = np.unique(Xm_ben, axis=0, return_inverse=True)
packed_b = np.packbits(uniq_b, axis=1)
U_b = len(packed_b)

parent = list(range(U_b))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


for b in range(len(fm)):
    byte_i, bit_i = b // 8, 7 - (b % 8)
    buckets = {}
    for v in range(U_b):
        key = packed_b[v].copy()
        key[byte_i] = key[byte_i] & (255 - (1 << bit_i))
        k = key.tobytes()
        if k in buckets:
            union(v, buckets[k])
        else:
            buckets[k] = v

hard_root_b = np.array([find(inv_b[i]) for i in range(len(Xm_ben))])
n_hard_groups = len(set(hard_root_b.tolist()))
print(
    f"benign exact groups={len(np.unique(inv_b))}, benign r1-hardened groups={n_hard_groups}",
    flush=True,
)

dfr, fr = load_ransmap_clean(drop_zero_var=True)
Xr_raw = dfr[fr].fillna(0.0).to_numpy(np.float64)
is_ben_r = dfr["label"].to_numpy() == 0
famr = dfr["family"].to_numpy()


def pair_no_inflate(a, b, rng, n):
    ia = rng.choice(len(a), n, replace=(len(a) < n))
    ib = rng.choice(len(b), n, replace=(len(b) < n))
    return a[ia], b[ib]


def oof_mlran_grouped(Xmb_tr, gids_tr, seed, n_splits=5):
    ug = np.unique(gids_tr)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_norm = np.zeros(len(Xmb_tr), dtype=np.float64)
    for fold_i, (f_tr_idx, _) in enumerate(kf.split(ug)):
        f_tr_g = set(ug[f_tr_idx])
        m_f = np.isin(gids_tr, list(f_tr_g))
        gm_f = GaussianMixture(
            n_components=K_MLRAN, covariance_type="diag", reg_covar=REG_COVAR,
            n_init=N_INIT, random_state=seed + fold_i
        ).fit(Xmb_tr[m_f])
        s_tr = -gm_f.score_samples(Xmb_tr[m_f])
        s_val = -gm_f.score_samples(Xmb_tr[~m_f])
        oof_norm[~m_f] = minmax_norm(s_tr, s_val)
    return oof_norm


def run_fusion(split_mode, seed):
    rng_split, rng_meta, rng_pair = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )

    if split_mode == "exact":
        ug = np.unique(inv_b)
        tr_g, _ = train_test_split(ug, test_size=TEST_SIZE, random_state=seed)
        mtr_b = np.isin(inv_b, tr_g)
        gids_tr = inv_b[mtr_b]
    else:
        ug = np.unique(hard_root_b)
        tr_g, _ = train_test_split(ug, test_size=TEST_SIZE, random_state=seed)
        mtr_b = np.isin(hard_root_b, tr_g)
        gids_tr = hard_root_b[mtr_b]

    Xmb_tr = Xm_ben[mtr_b].astype(np.float64)
    Xmb_te = Xm_ben[~mtr_b].astype(np.float64)
    gm_m = GaussianMixture(
        n_components=K_MLRAN, covariance_type="diag", reg_covar=REG_COVAR, n_init=N_INIT, random_state=seed
    ).fit(Xmb_tr)
    sm_tr = -gm_m.score_samples(Xmb_tr)
    sm_ben_te_n = minmax_norm(sm_tr, -gm_m.score_samples(Xmb_te))

    # stratify + StandardScaler on RanSMAP benign train
    idx_ben = np.where(is_ben_r)[0]
    tr_b, te_b = train_test_split(
        idx_ben, test_size=TEST_SIZE, random_state=seed, stratify=famr[idx_ben]
    )
    scaler_r = StandardScaler().fit(Xr_raw[tr_b])
    Xr = scaler_r.transform(Xr_raw)
    gm_r = GaussianMixture(
        n_components=K_RANSMAP, covariance_type="full", reg_covar=REG_COVAR, n_init=N_INIT, random_state=seed
    ).fit(Xr[tr_b])
    sr_tr = -gm_r.score_samples(Xr[tr_b])
    sr_ben_te_n = minmax_norm(sr_tr, -gm_r.score_samples(Xr[te_b]))

    # 5-fold OOF training calibration (n=600) + full 150 test evaluation
    sm_oof = oof_mlran_grouped(Xmb_tr, gids_tr, seed)
    sr_oof = oof_benign_norm_scores_ransmap(
        Xr_raw[tr_b], famr[tr_b], 5, seed, scale=True
    )
    n_calib = min(len(sm_oof), len(sr_oof))
    pa_c, pb_c = pair_no_inflate(sm_oof, sr_oof, rng_meta, n_calib)
    meta = GaussianMixture(
        META_K, covariance_type="full", reg_covar=META_REG_COVAR, n_init=N_INIT, random_state=seed
    ).fit(np.column_stack([pa_c, pb_c]))
    n_eval = min(len(sm_ben_te_n), len(sr_ben_te_n))

    out = {}
    for f in CANON_R:
        Xm_mal_f = Xm[(fam_m == f.lower()) & (ym == 1)].astype(np.float64)
        Xr_mal_f = Xr[(famr == f) & (~is_ben_r)]
        n = min(len(Xm_mal_f), len(Xr_mal_f))
        if n == 0:
            continue
        sm_mal_n = minmax_norm(sm_tr, -gm_m.score_samples(Xm_mal_f))
        sr_mal_n = minmax_norm(sr_tr, -gm_r.score_samples(Xr_mal_f))
        pa_m, pb_m = pair_no_inflate(sm_mal_n, sr_mal_n, rng_pair, n)
        pa_be, pb_be = pair_no_inflate(sm_ben_te_n, sr_ben_te_n, rng_pair, n_eval)
        s_mal = -meta.score_samples(np.column_stack([pa_m, pb_m]))
        s_ben = -meta.score_samples(np.column_stack([pa_be, pb_be]))
        yp = np.concatenate([np.ones(n), np.zeros(n_eval)])
        sp = np.concatenate([s_mal, s_ben])
        out[f] = float(roc_auc_score(yp, sp))
    return out


res = {"exact": {f: [] for f in CANON_R}, "hardened_r1": {f: [] for f in CANON_R}}
for seed in SEEDS:
    for mode in ("exact", "hardened_r1"):
        d = run_fusion(mode, seed)
        for f, v in d.items():
            res[mode][f].append(v)
    print(f"seed{seed} done", flush=True)

summary = {
    "description": __doc__.split("\n")[0],
    "per_family": {},
    "macro": {},
}
for mode in res:
    fam_means = {f: round(float(np.mean(v)), 4) for f, v in res[mode].items()}
    macro = round(float(np.mean([np.mean(v) for v in res[mode].values()])), 4)
    summary["per_family"][mode] = fam_means
    summary["macro"][mode] = macro
with open(OUT, "w") as f:
    json.dump(
        {
            **summary,
            "raw": {
                m: {f: [round(x, 4) for x in v] for f, v in res[m].items()}
                for m in res
            },
        },
        f,
        indent=2,
    )
print(json.dumps(summary, indent=2))
print("Saved ->", OUT)
