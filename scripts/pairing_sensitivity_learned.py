"""Bootstrap-pairing sensitivity of the fused AUC.

MLRan and RanSMAP malware samples are not aligned (different machines and
runs), so the fused evaluation pairs scores across channels by random
bootstrap sampling. This script measures how much the fused AUC depends on
the particular pairing, holding the fitted meta-GMM fixed per seed:
  for each of 10 seeds (fixed per-channel GMMs, fixed split, one meta-GMM
  fitted on a calibration-half pairing with the component count selected in
  learned_fusion.py):
    -> draw N_PAIRINGS = 100 random pairings of the evaluation-half benign
       and malware scores (the calibration half is never touched);
    -> compute the fused AUC for each pairing;
    -> report the within-seed standard deviation (pairing variance).
If pairing variance is small next to between-seed (model) variance, the
result does not hinge on the pairing draw (paper, Section 3.5).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import MLRAN_CSV, RANSMAP_CSV, RESULTS_DIR, SEEDS, CANONICAL_FAMILIES, K_MLRAN, K_RANSMAP, N_INIT, META_REG_COVAR, selected_meta_k  # noqa: E402

import hashlib
import json
import os
import sys


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from utils import fit_gm, minmax_norm

N_PAIRINGS = 100
META_K = selected_meta_k()


MLRAN_PATH = MLRAN_CSV
RANSMAP_PATH = RANSMAP_CSV
OUTPUT_DIR = RESULTS_DIR

MLRAN_EXCLUDE = {"label", "family", "sample_id", "sha256", "split", "STRING:labels", "dataset"}
RANSMAP_EXCLUDE = {"label", "family", "trace_path", "dataset"}

MLRAN_FAMILY_MAP = {"Conti": "conti", "Ryuk": "ryuk", "LockBit": "lockbit", "REvil": "revil", "WannaCry": "wannacry"}
RANSMAP_FAMILY_MAP = {"Conti": "Conti", "Ryuk": "Ryuk", "LockBit": "LockBit", "REvil": "REvil", "WannaCry": "WannaCry"}


def split_half(arr, rng):
    idx = rng.permutation(len(arr))
    h = len(arr) // 2
    return arr[idx[:h]], arr[idx[h:]]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MLRan...")
    mlran = pd.read_csv(MLRAN_PATH, index_col=0)
    mfc = [c for c in mlran.columns if c not in MLRAN_EXCLUDE]
    m_ben = mlran[mlran["label"] == 0].reset_index(drop=True)
    m_gids = np.array([hashlib.md5(r.astype(np.float32).tobytes()).hexdigest()
                        for r in m_ben[mfc].fillna(0).values])
    m_ben = m_ben.copy()
    m_ben["_gid"] = m_gids
    unique_gids = np.unique(m_gids)
    m_mal = {f: mlran[mlran["family"] == a][mfc].fillna(0.0).values
             for f, a in MLRAN_FAMILY_MAP.items()}

    print("Loading RanSMAP...")
    ransmap = pd.read_csv(RANSMAP_PATH)
    num_cols = set(ransmap.select_dtypes(include="number").columns)
    rfc = [c for c in ransmap.columns if c in num_cols and c not in RANSMAP_EXCLUDE]
    r_ben = ransmap[ransmap["label"] == 0].reset_index(drop=True)
    r_mal = {f: ransmap[ransmap["family"] == a][rfc].fillna(0.0).values
              for f, a in RANSMAP_FAMILY_MAP.items()}

    auc_grid: dict[str, list[list[float]]] = {f: [] for f in CANONICAL_FAMILIES}

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        rng_pair, rng_split, rng_meta = (np.random.default_rng(s)
                                          for s in np.random.SeedSequence(seed).spawn(3))

        tr_gids, _ = train_test_split(unique_gids, test_size=0.2, random_state=seed)
        mask = m_ben["_gid"].isin(set(tr_gids))
        Xmb_tr = m_ben[mask][mfc].fillna(0.0).values
        Xmb_te = m_ben[~mask][mfc].fillna(0.0).values

        Xrb_tr_raw, Xrb_te_raw = train_test_split(
            r_ben[rfc].fillna(0.0).values, test_size=0.2,
            random_state=seed, stratify=r_ben["family"].values,
        )
        sc = StandardScaler().fit(Xrb_tr_raw)
        Xrb_tr = sc.transform(Xrb_tr_raw)
        Xrb_te = sc.transform(Xrb_te_raw)

        gm_m = fit_gm(Xmb_tr, K_MLRAN, "diag", seed)
        gm_r = fit_gm(Xrb_tr, K_RANSMAP, "full", seed)

        sm_tr = -gm_m.score_samples(Xmb_tr)
        sr_tr = -gm_r.score_samples(Xrb_tr)

        sm_ben_te = minmax_norm(sm_tr, -gm_m.score_samples(Xmb_te))
        sr_ben_te = minmax_norm(sr_tr, -gm_r.score_samples(Xrb_te))

        # Leak-free calibration/eval split of the benign TEST pool (per
        # channel, independently), exactly as in learned_fusion.py.
        sm_ben_calib, sm_ben_eval = split_half(sm_ben_te, rng_split)
        sr_ben_calib, sr_ben_eval = split_half(sr_ben_te, rng_split)

        # Fit the meta-GMM ONCE per seed, on a calibration-half bootstrap
        # pairing, then hold it fixed while varying the pairing draw below.
        # This isolates pairing variance from meta-model-fit variance,
        # exactly as the original script holds gm_m/gm_r fixed per seed.
        n_calib_p = min(len(sm_ben_calib), len(sr_ben_calib))
        ia = rng_meta.choice(len(sm_ben_calib), max(n_calib_p, 200), replace=(len(sm_ben_calib) < 200))
        ib = rng_meta.choice(len(sr_ben_calib), max(n_calib_p, 200), replace=(len(sr_ben_calib) < 200))
        meta = GaussianMixture(META_K, covariance_type='full', reg_covar=META_REG_COVAR,
                                n_init=N_INIT, random_state=seed).fit(
            np.column_stack([sm_ben_calib[ia], sr_ben_calib[ib]]))

        n_ben_p = min(len(sm_ben_eval), len(sr_ben_eval))

        for fam in CANONICAL_FAMILIES:
            Xm = m_mal[fam]
            Xr = sc.transform(r_mal[fam])
            sm_mal = minmax_norm(sm_tr, -gm_m.score_samples(Xm))
            sr_mal = minmax_norm(sr_tr, -gm_r.score_samples(Xr))
            n_mal_p = min(len(sm_mal), len(sr_mal))

            pairing_aucs = []
            for _ in range(N_PAIRINGS):
                im = rng_pair.choice(len(sm_mal), n_mal_p, replace=(len(sm_mal) < n_mal_p))
                ir = rng_pair.choice(len(sr_mal), n_mal_p, replace=(len(sr_mal) < n_mal_p))
                ibn = rng_pair.choice(len(sm_ben_eval), n_ben_p, replace=(len(sm_ben_eval) < n_ben_p))
                ibrn = rng_pair.choice(len(sr_ben_eval), n_ben_p, replace=(len(sr_ben_eval) < n_ben_p))

                s_ben = -meta.score_samples(np.column_stack([sm_ben_eval[ibn], sr_ben_eval[ibrn]]))
                s_mal = -meta.score_samples(np.column_stack([sm_mal[im], sr_mal[ir]]))
                s_fused = np.concatenate([s_ben, s_mal])
                y = np.array([0] * n_ben_p + [1] * n_mal_p)
                try:
                    auc = float(roc_auc_score(y, s_fused))
                    pairing_aucs.append(auc)
                except Exception:
                    pass

            auc_grid[fam].append(pairing_aucs)
            mean_p = float(np.mean(pairing_aucs))
            std_p = float(np.std(pairing_aucs))
            print(f"  {fam:10s}  pairing AUC: {mean_p:.4f} +/- {std_p:.4f}  "
                  f"(range [{np.min(pairing_aucs):.4f},{np.max(pairing_aucs):.4f}])")

    print("\n\n=== SENSITIVITY SUMMARY (B_learned) ===")
    print(f"{'Family':<12}  {'Pairing std':>12}  {'Seed std':>10}  {'Ratio':>8}  "
          f"{'Mean AUC':>9}  {'Verdict'}")
    print("-" * 75)

    results = {}
    for fam in CANONICAL_FAMILIES:
        seed_means = [float(np.mean(s)) for s in auc_grid[fam]]
        seed_stds = [float(np.std(s, ddof=1)) for s in auc_grid[fam]]

        pairing_std = float(np.mean(seed_stds))
        seed_std = float(np.std(seed_means, ddof=1))
        overall_mean = float(np.mean(seed_means))
        ratio = pairing_std / (seed_std + 1e-8)

        verdict = "STABLE" if ratio < 0.5 else ("MODERATE" if ratio < 1.0 else "UNSTABLE")
        print(f"  {fam:<10}  {pairing_std:>12.5f}  {seed_std:>10.5f}  {ratio:>8.3f}  "
              f"{overall_mean:>9.4f}  {verdict}")

        results[fam] = {
            "mean_auc": round(overall_mean, 4),
            "pairing_std": round(pairing_std, 6),
            "seed_std": round(seed_std, 6),
            "pairing_to_seed_ratio": round(ratio, 4),
            "verdict": verdict,
            "per_seed_mean_auc": [round(m, 4) for m in seed_means],
            "per_seed_pairing_std": [round(s, 6) for s in seed_stds],
        }

    macro_pairing_std = float(np.mean([results[f]["pairing_std"] for f in CANONICAL_FAMILIES]))
    macro_seed_std = float(np.mean([results[f]["seed_std"] for f in CANONICAL_FAMILIES]))
    macro_ratio = macro_pairing_std / (macro_seed_std + 1e-8)
    print(f"\n  {'Macro':<10}  {macro_pairing_std:>12.5f}  {macro_seed_std:>10.5f}  "
          f"{macro_ratio:>8.3f}")

    out_path = OUTPUT_DIR / "pairing_sensitivity_learned.json"
    with open(out_path, "w") as f:
        json.dump({
            "description": (
                "Bootstrap pairing sensitivity for B_learned: for each seed "
                "(fixed per-channel GMMs + train/test split + one meta-GMM "
                f"fit on a calibration-half pairing), run {N_PAIRINGS} random "
                "bootstrap pairings of the eval-half benign scores and "
                "malware scores. Report within-seed std (pairing variance) "
                "vs between-seed std (model variance)."
            ),
            "n_seeds": len(SEEDS),
            "n_pairings_per_seed": N_PAIRINGS,
            "strategy": "B_learned",
            "meta_k": META_K,
            "per_family": results,
            "macro": {
                "pairing_std": round(macro_pairing_std, 6),
                "seed_std": round(macro_seed_std, 6),
                "ratio": round(macro_ratio, 4),
            },
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
