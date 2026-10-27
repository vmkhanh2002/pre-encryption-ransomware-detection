# Benign-Only Dual-Channel Ransomware Anomaly Fusion

Code for the SOICT 2026 paper

> M.-K. Vu, T.-P. Nguyen, D.-L. Vu. *Benign-Only Ransomware Detection via
> Dual-Channel Behavioral Anomaly Fusion: A Feasibility Study.* SOICT 2026,
> Springer CCIS.

Two Gaussian Mixture Models are trained on **benign data only** (no malware
labels): one on sandbox API/static-artifact features (MLRan), one on
hypervisor block-I/O metrics (RanSMAP). Their anomaly scores are fused by a
**learned density model** over the benign score cloud instead of a hand-picked
rule. The two public datasets were never recorded on the same host, so fused
scores are formed from bootstrap-paired samples and every fused result is a
feasibility study, not a measurement of a deployed detector.

## Main results (10 seeds)

| | Value |
|---|---|
| MLRan channel alone, AUC (64 families) | 0.9743 |
| RanSMAP channel alone, AUC (14 families) | 0.9960 |
| Fused, macro AUC (5 canonical families, LOFO) | 0.9989 |
| RanSMAP alone on the same benchmark | 0.9978 (all 5 families favour fusion; exact sign-flip test p = 0.0625, the smallest value 5 families allow) |
| TPR @ 1% FPR, fused vs. RanSMAP alone | 95.5% vs. 95.2% |
| Deploy-95 threshold, fused | 99.9% TPR at 9.5% macro FPR (3.5% with out-of-fold calibration) |

Fusion is **not** a significant accuracy gain on this saturated benchmark.
Its measured effect is under attack:

| Attack (feature space) | Fused macro AUC |
|---|---|
| None | 0.998 |
| Zero-knowledge throttling, up to 100x slowdown | never evades (I/O channel AUC >= 0.9978) |
| White-box on the I/O channel alone (that channel: 0.51) | 0.803; Deploy-95 TPR 99.7% -> 50.8% |
| Joint attack on both channels, static heuristic | 0.184 |
| Joint attack on both channels, coordinate descent | 0.185 |

The joint-attack values come from heuristic searches, not a certified worst
case. All attacks except throttling perturb feature vectors, not executable
ransomware. A Deploy-95 threshold does not transfer to an unseen CPU
(false-positive rate 8.0-26.6%).

**Correction (2026-10-27).** The first public version of this code had
three errors that affected the fused and adversarial results: Mahalanobis
distance used the covariance instead of its inverse; the fusion and attack
scripts did not standardize the RanSMAP features, although the paper says
they do; and the meta-GMM calibration resampled 75 scores into 200 with
replacement. The camera-ready paper reports the corrected numbers above.
The single-channel results (Table 2) were not affected.

## Layout

```
src/
  config.py                      paths and shared settings (single source)
  utils.py                       GMM fitting, data loading, normalization
  ano_models_eval.py             5 benign-only models x 2 channels (Table 2)
  ano_combined_eval.py           fixed-rule fusion reference
  ano_tpr_fpr01.py               TPR at 0.1% FPR
  ano_mlran_feature_ablation.py  MLRan feature-group ablation
scripts/
  learned_fusion.py              learned fusion, BIC-selected meta-GMM (Table 3)
  learned_fusion_significance.py family-blocked permutation test
  pairing_sensitivity_learned.py bootstrap-pairing variance
  channel_correlation_sensitivity.py  fused AUC vs. assumed channel correlation
  throttling_evasion.py          zero-knowledge throttling
  feature_space_adversarial.py, whitebox_adversarial.py   Level 2
  whitebox_mode_attack.py        Level 3, white-box (Table 4)
  whitebox_feature_freeze*.py    feature-freeze test and randomized control
  fused_under_attack_learned.py  fused system under the Level-3 attack
  joint_channel_adversarial*.py  joint two-channel attacks (v1, v2)
  cross_hardware_lofo.py         leave-one-CPU-out
  exact_fpr_counts.py, prevalence_ppv_analysis.py, hardened_fusion.py,
  near_duplicate_leakage.py, nested_k_selection.py,
  bootstrap_stability_final.py, bernoulli_vs_gmm.py      robustness checks
  generate_figures.py            figures, read from the result files
  build_ransmap_features.py      feature extraction from the RanSMAP archive
data/processed/                  place the two feature CSVs here (not in git)
```

## Setup

```bash
git clone https://github.com/vmkhanh2002/pre-encryption-ransomware-detection.git
cd pre-encryption-ransomware-detection
pip install -r requirements.txt
```

Obtain the datasets from their publishers and place the processed feature
tables in `data/processed/` (see `data/processed/README.md`).

## Reproduce

Run from any directory. Later scripts read results written by earlier ones,
so keep this order:

```bash
python src/ano_models_eval.py                     # per-channel models (Table 2)
python src/ano_combined_eval.py
python src/ano_tpr_fpr01.py
python src/ano_mlran_feature_ablation.py
python scripts/learned_fusion.py                  # fused system (Table 3)
python scripts/learned_fusion_significance.py
python scripts/pairing_sensitivity_learned.py
python scripts/channel_correlation_sensitivity.py
python scripts/fused_under_attack_learned.py
python scripts/hardened_fusion.py
python scripts/joint_channel_adversarial.py
python scripts/joint_channel_adversarial_v2.py
python scripts/prevalence_ppv_analysis.py
python scripts/bernoulli_vs_gmm.py
python scripts/bootstrap_stability_final.py
python scripts/nested_k_selection.py
python scripts/near_duplicate_leakage.py
python scripts/exact_fpr_counts.py
python scripts/feature_space_adversarial.py
python scripts/whitebox_adversarial.py
python scripts/whitebox_mode_attack.py            # Table 4
python scripts/whitebox_feature_freeze.py
python scripts/whitebox_feature_freeze_control.py
python scripts/throttling_evasion.py
python scripts/cross_hardware_lofo.py
python scripts/generate_figures.py                # Fig. 2
```

Results go to `experiments/ano_baseline/`, figures to `figures/`. All
randomness is seeded (seeds 0-9).

## Configuration

Paths and settings shared by several scripts (seeds, families, GMM component
counts, covariance types, regularization) are defined once in
`src/config.py`. Values computed by one script and used by another (for
example the meta-GMM component count selected by BIC) are read from its
result file, never typed in. Paths can be overridden with environment
variables:

| Variable | Default |
|---|---|
| `PERD_REPO_ROOT` | detected from the location of `src/config.py` |
| `PERD_MLRAN_CSV` | `data/processed/mlran_analysis_features.csv` |
| `PERD_RANSMAP_CSV` | `data/processed/ransmap_analysis_features.csv` |
| `PERD_RESULTS_DIR` | `experiments/ano_baseline/` |
| `PERD_FIGURES_DIR` | `figures/` |
| `RANSMAP_KAGGLE_ZIP` | required only for feature extraction |

## Citation

See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). MLRan and RanSMAP are distributed by their
authors under their own licenses and are not included here.
