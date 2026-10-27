"""Near-duplicate leakage check and hardened group split for MLRan.

Exact-hash grouping misses near-duplicates (vectors differing in 1-3 bits).
Using benign samples only, this script:
1. counts cross train/test pairs within a small Hamming radius under the
   exact-hash split (leakage estimate);
2. builds hardened groups by union-find over benign pairs within radius r
   (packed uint8 XOR + popcount over the unique benign vectors);
3. reruns the benign-only GMM AUC with hardened vs. exact-hash groups,
   scoring all malware samples.
"""
import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import K_MLRAN, N_INIT, REG_COVAR, RESULTS_DIR, SEEDS, TEST_SIZE  # noqa: E402
from utils import load_mlran  # noqa: E402

OUT = RESULTS_DIR / "near_duplicate_leakage.json"

df, feat_cols = load_mlran()
X = df[feat_cols].fillna(0).to_numpy(np.uint8)
y = df["label"].to_numpy()

X_ben = X[y == 0]
X_mal = X[y == 1]
N_ben = len(X_ben)
N_mal = len(X_mal)

# Unique BENIGN vectors + mapping benign row -> unique id
uniq_b, inv_b = np.unique(X_ben, axis=0, return_inverse=True)
U_b = len(uniq_b)
print(f"benign_rows={N_ben} unique_benign_vectors={U_b} malware_rows={N_mal}", flush=True)

packed_b = np.packbits(uniq_b, axis=1)


def hamming_matrix_chunked(packed, chunk=200):
    U = len(packed)
    D = np.zeros((U, U), dtype=np.uint16)
    for i in range(0, U, chunk):
        xi = packed[i : i + chunk]
        for j in range(i, U, chunk):
            xj = packed[j : j + chunk]
            xor = xi[:, None, :] ^ xj[None, :, :]
            d = np.unpackbits(xor, axis=2).sum(axis=2)
            D[i : i + len(xi), j : j + len(xj)] = d
            if i != j:
                D[j : j + len(xj), i : i + len(xi)] = d.T
    return D


print("computing pairwise Hamming on unique BENIGN vectors...", flush=True)
D_b = hamming_matrix_chunked(packed_b)


class UF:
    def __init__(s, n):
        s.p = list(range(n))

    def find(s, a):
        while s.p[a] != a:
            s.p[a] = s.p[s.p[a]]
            a = s.p[a]
        return a

    def union(s, a, b):
        s.p[s.find(a)] = s.find(b)


def grouped_split(hard_groups, seed):
    ug = np.unique(hard_groups)
    tr, _ = train_test_split(ug, test_size=TEST_SIZE, random_state=seed)
    return np.isin(hard_groups, tr)


res = {
    "description": __doc__.split("\n")[0],
    "n_benign_rows": int(N_ben),
    "n_unique_benign_vectors": int(U_b),
    "n_malware_rows": int(N_mal),
    "radii": {},
}

for r in [0, 1, 2, 3, 5]:
    uf = UF(U_b)
    if r > 0:
        ii, jj = np.where((D_b <= r) & (D_b > 0))
        for a, b in zip(ii, jj):
            uf.union(int(a), int(b))
        n_pairs = len(ii) // 2
    else:
        n_pairs = 0
    hard_b = np.array([uf.find(inv_b[i]) for i in range(N_ben)])
    n_hard = len(np.unique(hard_b))

    leak_fracs = []
    aucs_exact, aucs_hard = [], []
    for seed in SEEDS:
        # Exact-hash split on benign unique vectors
        ug_ex = np.unique(inv_b)
        tr_e, _ = train_test_split(ug_ex, test_size=TEST_SIZE, random_state=seed)
        mtr_e = np.isin(inv_b, tr_e)
        mte_e = ~mtr_e

        # Leakage: min Hamming distance from each benign test unique-vector to any benign train unique-vector
        te_u = np.unique(inv_b[mte_e])
        tr_u = np.unique(inv_b[mtr_e])
        mind = D_b[te_u][:, tr_u].min(axis=1)
        leak_fracs.append(float((mind <= r).mean()))

        # AUC with hardened benign groups (evaluated against 100% malware)
        mtr_h = grouped_split(hard_b, seed)
        Xtr_ben_h = X_ben[mtr_h].astype(np.float32)
        Xte_ben_h = X_ben[~mtr_h].astype(np.float32)
        gm = GaussianMixture(
            n_components=K_MLRAN, covariance_type="diag", reg_covar=REG_COVAR,
            n_init=N_INIT, random_state=seed
        ).fit(Xtr_ben_h)
        X_eval_h = np.vstack([Xte_ben_h, X_mal.astype(np.float32)])
        y_eval_h = np.concatenate([np.zeros(len(Xte_ben_h)), np.ones(N_mal)])
        aucs_hard.append(float(roc_auc_score(y_eval_h, -gm.score_samples(X_eval_h))))

        # Baseline AUC with exact benign groups (evaluated against 100% malware)
        Xtr_ben_e = X_ben[mtr_e].astype(np.float32)
        Xte_ben_e = X_ben[mte_e].astype(np.float32)
        gm2 = GaussianMixture(
            n_components=K_MLRAN, covariance_type="diag", reg_covar=REG_COVAR,
            n_init=N_INIT, random_state=seed
        ).fit(Xtr_ben_e)
        X_eval_e = np.vstack([Xte_ben_e, X_mal.astype(np.float32)])
        y_eval_e = np.concatenate([np.zeros(len(Xte_ben_e)), np.ones(N_mal)])
        aucs_exact.append(float(roc_auc_score(y_eval_e, -gm2.score_samples(X_eval_e))))

    res["radii"][str(r)] = {
        "unique_benign_groups": int(n_hard),
        "merge_pairs_at_r": int(n_pairs),
        "mean_leak_frac_train_prox": round(float(np.mean(leak_fracs)), 4),
        "auc_exact_mean": round(float(np.mean(aucs_exact)), 4),
        "auc_hardened_mean": round(float(np.mean(aucs_hard)), 4),
        "auc_delta": round(float(np.mean(aucs_hard) - np.mean(aucs_exact)), 4),
    }
    print(f"r={r}:", res["radii"][str(r)], flush=True)

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print("Saved ->", OUT)
