"""
MLRan feature-composition ablation: standalone-channel AUC when restricted
to API:* only, vs STRING:*+SIGNATURE:* only, vs the full 482-dim vector.
Same GMM(K=3, diag) + group-based-split + 5-family-LOFO protocol as the
other MLRan-channel results.
"""

from config import MLRAN_CSV, RESULTS_DIR, SEEDS, CANONICAL_FAMILIES, K_MLRAN  # noqa: E402
import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from utils import fit_gm


MLRAN_PATH = MLRAN_CSV
OUTPUT_DIR = RESULTS_DIR
MLRAN_EXCLUDE = {"label", "family", "sample_id", "sha256", "split", "STRING:labels", "dataset"}
MLRAN_FAMILY_MAP = {"Conti": "conti", "Ryuk": "ryuk", "LockBit": "lockbit",
                     "REvil": "revil", "WannaCry": "wannacry"}


def score(gm, X):
    return -gm.score_samples(X)


def main():
    mlran = pd.read_csv(MLRAN_PATH, index_col=0)
    all_feat = [c for c in mlran.columns if c not in MLRAN_EXCLUDE]

    subsets = {
        "full_482": all_feat,
        "api_only_63": [c for c in all_feat if c.startswith("API:")],
        "string_signature_242": [c for c in all_feat if c.startswith("STRING:") or c.startswith("SIGNATURE:")],
    }
    for name, cols in subsets.items():
        print(f"{name}: {len(cols)} features")

    m_ben_raw = mlran[mlran["label"] == 0].reset_index(drop=True)
    m_mal = {f: mlran[mlran["family"] == a].reset_index(drop=True) for f, a in MLRAN_FAMILY_MAP.items()}

    results = {name: {f: [] for f in CANONICAL_FAMILIES} for name in subsets}

    for subset_name, cols in subsets.items():
        m_gids = np.array([hashlib.md5(row.astype(np.float32).tobytes()).hexdigest()
                            for row in m_ben_raw[cols].fillna(0).values])
        m_ben = m_ben_raw.copy()
        m_ben["_gid"] = m_gids
        m_unique_gids = np.unique(m_gids)

        for seed in SEEDS:
            tr_gids, te_gids = train_test_split(m_unique_gids, test_size=0.2, random_state=seed)
            m_tr_mask = m_ben["_gid"].isin(set(tr_gids))
            Xtr = m_ben[m_tr_mask][cols].fillna(0.0).values
            Xte = m_ben[~m_tr_mask][cols].fillna(0.0).values

            gm = fit_gm(Xtr, n_components=K_MLRAN, covariance_type="diag", seed=seed)

            for family in CANONICAL_FAMILIES:
                Xmal = m_mal[family][cols].fillna(0.0).values
                y = np.array([0] * len(Xte) + [1] * len(Xmal))
                s = np.concatenate([score(gm, Xte), score(gm, Xmal)])
                results[subset_name][family].append(float(roc_auc_score(y, s)))

        print(f"  {subset_name} done")

    summary = {}
    for name in subsets:
        macro = [np.mean(results[name][f]) for f in CANONICAL_FAMILIES]
        all_runs = [v for f in CANONICAL_FAMILIES for v in results[name][f]]
        summary[name] = {
            "n_features": len(subsets[name]),
            "macro_mean_auc": round(float(np.mean(macro)), 4),
            "macro_std_auc": round(float(np.std(all_runs, ddof=1)), 4),
            "per_family_mean_auc": {f: round(float(np.mean(results[name][f])), 4) for f in CANONICAL_FAMILIES},
        }
    print(json.dumps(summary, indent=2))

    out_path = OUTPUT_DIR / "ano_mlran_feature_ablation.json"
    with open(out_path, "w") as f:
        json.dump({
            "description": "MLRan standalone-channel AUC by feature subset, GMM(K=3,diag), 5-family LOFO x 10 seeds, group-based split per subset.",
            "n_seeds": len(SEEDS),
            "results": summary,
        }, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
