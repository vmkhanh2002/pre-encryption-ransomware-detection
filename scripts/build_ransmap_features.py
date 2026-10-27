"""Build ransmap_analysis_features.csv from the RanSMAP dataset archive.

Streams ata_read/ata_write CSVs straight out of the archive (no extraction
to disk) across ALL hardware configs: dataset/original/*, dataset/extra/*,
dataset/variants/*. Produces 34 continuous block-I/O metrics per trace:
17 per direction (read/write): io_count, total_bytes, mean/std/min/max
size, unique_sectors, sector_range, duration_s, io_per_s, bytes_per_s,
frac_4k, sequential_frac, mean/std/max of the per-request entropy value
("weight"; ata_read.csv stores -1, clipped to 0, so the read versions are
constant), distinct sizes. Columns that are constant over all benign traces
(read entropy, maximum request size) are kept in the table and dropped at
load time by utils.load_ransmap_clean.

Data source: RanSMAP is published by Hirano & Kobayashi, "RanSMAP: Open
Dataset of Ransomware Storage and Memory Access Patterns for Creating Deep
Learning Based Ransomware Detectors", Computers & Security 150 (2025)
104202, DOI 10.1016/j.cose.2024.104202. Obtain the archive from the
dataset's official distribution referenced in that paper and point this
script at it via the RANSMAP_KAGGLE_ZIP environment variable, e.g.:

    RANSMAP_KAGGLE_ZIP=/path/to/ransmap.zip python scripts/build_ransmap_features.py

The script stops with a message if the variable is unset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import RANSMAP_CSV, ransmap_kaggle_zip  # noqa: E402
import os, sys, zipfile
import numpy as np
import pandas as pd

ZIP = str(ransmap_kaggle_zip())
OUT = RANSMAP_CSV
BENIGN = {'Firefox', 'Idle', 'Office', 'SDelete', 'Zip'}

if not os.path.isfile(ZIP):
    sys.exit(
        f"RanSMAP archive not found at: {ZIP}\n"
        "Set RANSMAP_KAGGLE_ZIP to the path of the RanSMAP dataset archive "
        "(see the module docstring for the dataset citation/DOI).")

def series_stats(ts, lba, size, w):
    n = len(size)
    dur = float(ts.max() - ts.min()) if n else 0.0
    wc = np.clip(w, 0, None)
    return [
        float(n),
        float(size.sum()),
        float(size.mean()) if n else 0.0,
        float(size.std(ddof=1)) if n > 1 else 0.0,
        float(size.max()) if n else 0.0,
        float(size.min()) if n else 0.0,
        int(pd.unique(lba).size),
        float(lba.max() - lba.min()) if n else 0.0,
        dur,
        float(n / dur) if dur > 0 else 0.0,
        float(size.sum() / dur) if dur > 0 else 0.0,
        float((size == 4096).mean()) if n else 0.0,
        float((np.diff(np.sort(lba)) == 8).mean()) if n > 1 else 0.0,
        float(wc.mean()) if n else 0.0,
        float(wc.std(ddof=1)) if n > 1 else 0.0,
        float(wc.max()) if n else 0.0,
        int(pd.unique(size).size),
    ]

SUFFIX = ['io_count','total_bytes','mean_size','std_size','max_size','min_size',
          'unique_sectors','sector_range','duration_s','io_per_s','bytes_per_s',
          'frac_4k','sequential_frac','mean_weight','std_weight','max_weight',
          'distinct_sizes']
COLNAMES = (['label','family','trace_path','dataset','cpu','ram']
            + ['r_'+s for s in SUFFIX] + ['w_'+s for s in SUFFIX])

zf = zipfile.ZipFile(ZIP)
members = [n for n in zf.namelist()
           if ('ata_read.csv' in n or 'ata_write.csv' in n)
           and any(f'/{seg}/' in n for seg in ('original', 'extra', 'variants'))]
# group by trace dir
from collections import defaultdict
traces = defaultdict(dict)
for m in members:
    d, f = os.path.split(m)
    traces[d][f] = m

print(f'traces found: {len(traces)}', flush=True)

records = []
for i, (d, files) in enumerate(sorted(traces.items())):
    parts = d.split('/')          # RanSMAP/dataset/<split>/CPU/RAM/APP/TRACE
    split, cpu, ram, app = parts[2], parts[3], parts[4], parts[5]
    rec = {
        'label': 0 if app in BENIGN else 1,
        'family': app,
        'trace_path': f'{split}/{cpu}/{ram}/{app}/{parts[5]}',
        'dataset': split, 'cpu': cpu, 'ram': ram,
    }
    for rw, fname in (('r','ata_read.csv'), ('w','ata_write.csv')):
        vals = [0.0]*17
        if fname in files:
            try:
                df = pd.read_csv(zf.open(files[fname]), header=None,
                                 usecols=[0,2,3,4], names=['ts','lba','sz','w'],
                                 skipinitialspace=True).dropna()
                if len(df):
                    vals = series_stats(df['ts'].to_numpy(), df['lba'].to_numpy(),
                                        df['sz'].to_numpy(), df['w'].to_numpy())
            except Exception as e:
                print('  ERR', files[fname], e, flush=True)
        rec.update(zip([rw+'_'+s for s in SUFFIX], vals))
    records.append(rec)
    if (i+1) % 200 == 0:
        print(f'{i+1}/{len(traces)} done', flush=True)

df = pd.DataFrame(records, columns=COLNAMES)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
print('written:', OUT)
print('rows:', len(df), '| cols:', df.shape[1])
print('labels:', df['label'].value_counts().to_dict())
print('families:', sorted(df['family'].unique()))
