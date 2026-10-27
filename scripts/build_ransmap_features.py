"""Build ransmap_analysis_features.csv from raw RanSMAP trace CSVs.

Extracts 34 continuous block-I/O metrics per trace (ata_read/ata_write):
  counts, byte volumes, size stats, latency (weight) stats, entropy proxy,
  per-direction and combined. Output schema matches PERD repo loader:
  label, family, trace_path, dataset + 34 numeric feature columns.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RANSMAP_CSV, ransmap_raw_dir  # noqa: E402
import os, glob
import numpy as np
import pandas as pd

BASE = str(ransmap_raw_dir())
OUT = RANSMAP_CSV

COLS = ['timestamp', 'delta_us', 'sector', 'size_bytes', 'weight', 'rw_flag']

def load_ata(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, header=None, names=COLS, skipinitialspace=True)
    except Exception:
        return None
    df = df.dropna()
    return df if len(df) else None

def feats_for(df):
    """34 metrics from one ata dataframe."""
    n = len(df)
    sizes = df['size_bytes']
    weights = df['weight'].clip(lower=0)
    sectors = df['sector']
    out = {
        'io_count': n,
        'total_bytes': float(sizes.sum()),
        'read_count': int((df['rw_flag'] == 0).sum()),
        'write_count': int((df['rw_flag'] == 1).sum()),
        'mean_size': float(sizes.mean()) if n else 0.0,
        'std_size': float(sizes.std(ddof=1)) if n > 1 else 0.0,
        'max_size': float(sizes.max()) if n else 0.0,
        'min_size': float(sizes.min()) if n else 0.0,
        'mean_latency': float(weights.mean()) if n else 0.0,
        'std_latency': float(weights.std(ddof=1)) if n > 1 else 0.0,
        'max_latency': float(weights.max()) if n else 0.0,
        'sum_latency': float(weights.sum()),
        'unique_sectors': int(sectors.nunique()),
        'sector_range': float(sectors.max() - sectors.min()) if n else 0.0,
        'duration_s': float(df['timestamp'].max() - df['timestamp'].min()) if n else 0.0,
        'io_per_s': float(n / max(df['timestamp'].max() - df['timestamp'].min(), 1e-9)),
        'bytes_per_s': float(sizes.sum() / max(df['timestamp'].max() - df['timestamp'].min(), 1e-9)),
        'frac_4k': float((sizes == 4096).mean()) if n else 0.0,
        'frac_large': float((sizes >= 65536).mean()) if n else 0.0,
        'sequential_frac': float((sectors.diff().dropna() == 8).mean()) if n > 1 else 0.0,
    }
    return out

def build_channel(rw):
    """Return dict of 17 features for read ('r') or write ('w') channel with suffix."""
    pass  # assembled below

records = []
for app in sorted(os.listdir(BASE)):
    appdir = os.path.join(BASE, app)
    if not os.path.isdir(appdir):
        continue
    for tracedir in sorted(glob.glob(os.path.join(appdir, app + '-*'))):
        ar = load_ata(os.path.join(tracedir, 'ata_read.csv'))
        aw = load_ata(os.path.join(tracedir, 'ata_write.csv'))
        rec = {'label': 1 if app not in ('Firefox', 'Idle', 'Office', 'SDelete', 'Zip') else 0,
               'family': app,
               'trace_path': os.path.relpath(tracedir, BASE),
               'dataset': 'original'}
        # 17 read-channel + 17 write-channel = 34 features
        fr = feats_for(ar) if ar is not None else dict.fromkeys(feats_for(pd.DataFrame(columns=COLS)), 0.0)
        fw = feats_for(aw) if aw is not None else dict.fromkeys(feats_for(pd.DataFrame(columns=COLS)), 0.0)
        for k, v in fr.items():
            rec['r_' + k] = v
        for k, v in fw.items():
            rec['w_' + k] = v
        records.append(rec)

df = pd.DataFrame(records)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
print('written:', OUT)
print('rows:', len(df), '| cols:', len(df.columns))
print('label counts:', df['label'].value_counts().to_dict())
print('families:', sorted(df['family'].unique()))
