"""TPR at 0.1% FPR for the two standalone channels and the fixed product rule.

5 canonical families x 10 seeds LOFO, same protocol as ano_combined_eval.py.
The paper reports the standalone-channel values and explains why a fused
value at this operating point is not meaningful for these test-set sizes
(Section 4.2).
"""

from config import CANONICAL_FAMILIES, K_MLRAN, K_RANSMAP, MLRAN_CSV, RANSMAP_CSV, RESULTS_DIR, SEEDS, TEST_SIZE  # noqa: E402
import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

from utils import fit_gm, minmax_norm


MLRAN_PATH = MLRAN_CSV
RANSMAP_PATH = RANSMAP_CSV
OUTPUT_DIR = RESULTS_DIR

MLRAN_EXCLUDE = {"label", "family", "sample_id", "sha256", "split", "STRING:labels", "dataset"}
RANSMAP_EXCLUDE = {"label", "family", "trace_path", "dataset"}
MLRAN_FAMILY_MAP = {f: f.lower() for f in CANONICAL_FAMILIES}
RANSMAP_FAMILY_MAP = {f: f for f in CANONICAL_FAMILIES}


def score(gm, X):
    return -gm.score_samples(X)


def tpr_at_fpr(y, s, target_fpr):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target_fpr, fpr, tpr))


def pair_scores(s_a, s_b, rng, n):
    ia = rng.choice(len(s_a), n, replace=(len(s_a) < n))
    ib = rng.choice(len(s_b), n, replace=(len(s_b) < n))
    return s_a[ia], s_b[ib]


def main():
    mlran = pd.read_csv(MLRAN_PATH, index_col=0)
    mfc = [c for c in mlran.columns if c not in MLRAN_EXCLUDE]
    m_ben_raw = mlran[mlran["label"] == 0].reset_index(drop=True)
    m_gids = np.array([hashlib.md5(row.astype(np.float32).tobytes()).hexdigest()
                        for row in m_ben_raw[mfc].fillna(0).values])
    m_ben = m_ben_raw.copy()
    m_ben["_gid"] = m_gids
    m_unique_gids = np.unique(m_gids)
    m_mal = {f: mlran[mlran["family"] == a].reset_index(drop=True) for f, a in MLRAN_FAMILY_MAP.items()}

    ransmap = pd.read_csv(RANSMAP_PATH)
    num_cols = set(ransmap.select_dtypes(include="number").columns)
    rfc = [c for c in ransmap.columns if c in num_cols and c not in RANSMAP_EXCLUDE]
    r_ben = ransmap[ransmap["label"] == 0].reset_index(drop=True)
    r_mal = {f: ransmap[ransmap["family"] == a].reset_index(drop=True) for f, a in RANSMAP_FAMILY_MAP.items()}

    keys = ["A_mlran", "A_ransmap", "B_product"]
    runs = {k: [] for k in keys}

    for seed in SEEDS:
        rng_b = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])

        tr_gids, te_gids = train_test_split(m_unique_gids, test_size=TEST_SIZE, random_state=seed)
        m_tr_mask = m_ben["_gid"].isin(set(tr_gids))
        Xmb_tr = m_ben[m_tr_mask][mfc].fillna(0.0).values
        Xmb_te = m_ben[~m_tr_mask][mfc].fillna(0.0).values

        Xrb_tr_raw, Xrb_te_raw = train_test_split(
            r_ben[rfc].fillna(0.0).values, test_size=TEST_SIZE,
            random_state=seed, stratify=r_ben["family"].values)
        sc = StandardScaler().fit(Xrb_tr_raw)
        Xrb_tr, Xrb_te = sc.transform(Xrb_tr_raw), sc.transform(Xrb_te_raw)

        gm_m = fit_gm(Xmb_tr, n_components=K_MLRAN, covariance_type="diag", seed=seed)
        gm_r = fit_gm(Xrb_tr, n_components=K_RANSMAP, covariance_type="full", seed=seed)
        sm_tr, sr_tr = score(gm_m, Xmb_tr), score(gm_r, Xrb_tr)

        for family in CANONICAL_FAMILIES:
            Xmmal = m_mal[family][mfc].fillna(0.0).values
            Xrmal = sc.transform(r_mal[family][rfc].fillna(0.0).values)

            ymt = np.array([0] * len(Xmb_te) + [1] * len(Xmmal))
            yrt = np.array([0] * len(Xrb_te) + [1] * len(Xrmal))
            runs["A_mlran"].append(tpr_at_fpr(ymt, np.concatenate([score(gm_m, Xmb_te), score(gm_m, Xmmal)]), 0.001))
            runs["A_ransmap"].append(tpr_at_fpr(yrt, np.concatenate([score(gm_r, Xrb_te), score(gm_r, Xrmal)]), 0.001))

            sm_ben_norm = minmax_norm(sm_tr, score(gm_m, Xmb_te))
            sr_ben_norm = minmax_norm(sr_tr, score(gm_r, Xrb_te))
            sm_mal_norm = minmax_norm(sm_tr, score(gm_m, Xmmal))
            sr_mal_norm = minmax_norm(sr_tr, score(gm_r, Xrmal))

            n_mal_p = min(len(Xmmal), len(Xrmal))
            n_ben_p = min(len(Xmb_te), len(Xrb_te))
            pa_m, pb_m = pair_scores(sm_mal_norm, sr_mal_norm, rng_b, n_mal_p)
            pa_b, pb_b = pair_scores(sm_ben_norm, sr_ben_norm, rng_b, n_ben_p)
            y = np.array([0] * n_ben_p + [1] * n_mal_p)
            s_fused = np.concatenate([pa_b * pb_b, pa_m * pb_m])
            runs["B_product"].append(tpr_at_fpr(y, s_fused, 0.001))

        print(f"seed {seed} done")

    summary = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v, ddof=1)), 4),
                    "ci95_lo": round(float(np.percentile(v, 2.5)), 4),
                    "ci95_hi": round(float(np.percentile(v, 97.5)), 4)}
               for k, v in runs.items()}
    print(json.dumps(summary, indent=2))

    out_path = OUTPUT_DIR / "ano_tpr_fpr01_results.json"
    with open(out_path, "w") as f:
        json.dump({"description": "TPR@0.1%FPR, 5 canonical families x 10 seeds LOFO.",
                    "n_runs_per_key": len(runs["B_product"]), "results": summary}, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
