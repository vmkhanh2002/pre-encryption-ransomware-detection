# data/processed/

Place the two processed feature tables here (not tracked by git):

- `mlran_analysis_features.csv`: 482 binary Cuckoo sandbox features
  (API, registry, file, string and signature indicators) from MLRan:
  F. C. Onwuegbuche, S. O. Adelodun, A. D. Jurcut, L. Pasquale,
  *MLRan: A behavioural dataset for ransomware analysis and detection*,
  Journal of Network and Computer Applications 250, 104475 (2026),
  doi:10.1016/j.jnca.2026.104475.
- `ransmap_analysis_features.csv`: 34 continuous hypervisor block-I/O
  metrics extracted from RanSMAP with `scripts/build_ransmap_features_full.py`:
  M. Hirano, R. Kobayashi, *RanSMAP: Open dataset of ransomware storage and
  memory access patterns for creating deep learning based ransomware
  detectors*, Computers & Security 150, 104202 (2025),
  doi:10.1016/j.cose.2024.104202.

Other locations can be used through the `PERD_MLRAN_CSV` and
`PERD_RANSMAP_CSV` environment variables (see `src/config.py`).
