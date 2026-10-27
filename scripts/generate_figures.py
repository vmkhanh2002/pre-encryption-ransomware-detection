#!/usr/bin/env python3
"""
generate_figures.py - Figures for the paper, drawn only from result files.

Every plotted value is read from the JSON results written by the other
scripts (run them first, order in README.md); nothing is typed in.

Requirements: matplotlib >= 3.5, numpy
Usage: python scripts/generate_figures.py

Outputs (config.FIGURES_DIR, default figures/):
  - fig_adversarial_cascade.{pdf,png}        (paper Fig. 2)
  - fig_channel_model_comparison.{pdf,png}   (Table 2 as a chart; not in the paper)
  - fig_correlation_sensitivity.{pdf,png}    (not in the paper)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(next(p / "src" for p in Path(__file__).resolve().parents
                            if (p / "src" / "config.py").is_file())))
from config import FIGURES_DIR, load_result  # noqa: E402
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -- Style configuration ----------------------------------------------------
# Color-blind friendly palette (IBM Design Library)
COLORS = {
    'blue':   '#648FFF',
    'orange': '#DC267F',  # magenta-pink, but distinct
    'yellow': '#FE6100',
    'purple': '#785EF0',
    'teal':   '#009E73',
    'red':    '#D55E00',
    'gray':   '#999999',
    'black':  '#000000',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT_DIR = FIGURES_DIR
os.makedirs(OUT_DIR, exist_ok=True)
_load = load_result


# -- Figure 1: Per-channel anomaly-model comparison (Table 1, visualised) ---
def fig_channel_model_comparison():
    d = _load('ano_models_results.json')
    models = ['IsolationForest', 'OneClassSVM', 'LOF', 'GaussianMixture', 'SGDOneClassSVM']
    labels = ['IsolationForest', 'OneClassSVM', 'LOF', 'GaussianMixture\n(selected)', 'SGDOneClassSVM']

    mlran_auc = [d['ano_mlran'][m]['mean_auc'] for m in models]
    mlran_std = [d['ano_mlran'][m]['std_auc'] for m in models]
    ransmap_auc = [d['ano_ransmap'][m]['mean_auc'] for m in models]
    ransmap_std = [d['ano_ransmap'][m]['std_auc'] for m in models]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, mlran_auc, width, yerr=mlran_std, capsize=3,
           color=COLORS['blue'], edgecolor='white', linewidth=0.5, label='MLRan (API)')
    ax.bar(x + width / 2, ransmap_auc, width, yerr=ransmap_std, capsize=3,
           color=COLORS['teal'], edgecolor='white', linewidth=0.5, label='RanSMAP (I/O)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('AUC (10 seeds, mean $\\pm$ std)')
    ax.set_ylim(0, 1.08)
    ax.axhline(y=1.0, color=COLORS['gray'], linestyle=':', linewidth=0.6, alpha=0.5)
    ax.legend(loc='lower center', ncol=2, fontsize=7, framealpha=0.9)
    ax.set_title('Benign-Only Anomaly Model Comparison, Per Channel')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_channel_model_comparison.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_channel_model_comparison.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved fig_channel_model_comparison (GMM MLRan={mlran_auc[3]:.4f}, '
          f'GMM RanSMAP={ransmap_auc[3]:.4f})')


# -- Figure 2: Adversarial-robustness cascade (the paper's central claim) --
def fig_adversarial_cascade():
    throttle = _load('throttling_evasion.json')
    fsa = _load('feature_space_adversarial.json')
    whitebox = _load('whitebox_mode_attack.json')
    l3_fused = _load('fused_under_attack_learned.json')
    joint_v1 = _load('joint_channel_adversarial.json')
    joint_v2 = _load('joint_channel_adversarial_v2.json')

    clean = joint_v1['summary']['macro_auc_clean']
    throttle_worst = min(v['auc_mean'] for v in throttle['summary'].values())
    l2_benign_mean = fsa['featurespace_auc_mean']
    l3_ransmap_alone = whitebox['summary']['nearest_mode_attack']['post_attack_auc_mean']
    l3_fused_auc = l3_fused['summary']['b_learned_auc_attacked_mean']
    joint_v1_auc = joint_v1['summary']['macro_auc_attackB_joint_worst_case']
    v2_rounds = joint_v2['summary']['macro_auc_by_round']
    joint_v2_auc = v2_rounds[max(v2_rounds, key=int)]  # final round of the search

    stages = ['Clean', 'Throttling\n(0-100x)', 'Benign-mean\ninterpolation',
              'White-box,\nRanSMAP alone', 'White-box,\nfused', 'Joint attack\n(heuristic)',
              'Joint attack\n(coord. descent)']
    values = [clean, throttle_worst, l2_benign_mean, l3_ransmap_alone,
              l3_fused_auc, joint_v1_auc, joint_v2_auc]
    colors = [COLORS['teal'], COLORS['teal'], COLORS['teal'],
              COLORS['orange'], COLORS['blue'], COLORS['red'], COLORS['red']]

    fig, ax = plt.subplots(figsize=(8.4, 3.3))
    x = np.arange(len(stages))
    ax.bar(x, values, color=colors, edgecolor='white', linewidth=0.5, width=0.65)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.025, f'{v:.3f}', ha='center', fontsize=9.5)

    ax.axhline(y=0.5, color=COLORS['gray'], linestyle='--', linewidth=1.0)
    ax.text(len(stages) - 0.6, 0.55, 'random guess (0.5)', fontsize=9,
            color=COLORS['gray'], ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=8.8)
    ax.set_ylabel('Macro AUC', fontsize=10.5)
    ax.set_ylim(0, 1.18)
    ax.tick_params(axis='y', labelsize=9)
    ax.set_title('Fused System Under Escalating Adversary Knowledge', fontsize=11.5)

    legend_handles = [
        mpatches.Patch(color=COLORS['teal'], label='No/low knowledge'),
        mpatches.Patch(color=COLORS['orange'], label='White-box, single channel'),
        mpatches.Patch(color=COLORS['blue'], label='White-box, fused (single-channel attack)'),
        mpatches.Patch(color=COLORS['red'], label='White-box, fused (joint two-channel attack)'),
    ]
    # below the axes, so no bar label is hidden
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.30),
              ncol=2, fontsize=7.8, frameon=False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_adversarial_cascade.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_adversarial_cascade.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved fig_adversarial_cascade (clean={clean:.4f} -> joint_v1={joint_v1_auc:.4f}, '
          f'joint_v2={joint_v2_auc:.4f})')


# -- Figure 3: Fused-AUC sensitivity to the channel-independence assumption -
def fig_correlation_sensitivity():
    d = _load('channel_correlation_sensitivity.json')['summary']
    keys = sorted(d.keys(), key=float)
    realized = [d[k]['realized_spearman_rho_mean'] for k in keys]
    aucs = [d[k]['macro_auc'] for k in keys]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(realized, aucs, marker='o', color=COLORS['purple'], linewidth=1.6, markersize=6)
    ax.set_xlabel('Realized inter-channel correlation ($\\rho$)')
    ax.set_ylabel('Fused macro AUC')
    lo, hi = min(aucs), max(aucs)
    pad = max(0.002, (hi - lo) * 0.6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.axvline(x=0, color=COLORS['gray'], linestyle=':', linewidth=0.8)
    ax.text(0.02, hi + pad * 0.5, 'independence\nbaseline', fontsize=6.2, color=COLORS['gray'])
    ax.set_title(f'Fused AUC vs. Assumed Channel Correlation\n(spread = {hi - lo:.4f})')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_correlation_sensitivity.pdf'))
    fig.savefig(os.path.join(OUT_DIR, 'fig_correlation_sensitivity.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved fig_correlation_sensitivity (AUC range [{lo:.4f}, {hi:.4f}])')


if __name__ == '__main__':
    print('Generating figures for the paper...')
    print(f'Output directory: {OUT_DIR}')
    print()

    fig_channel_model_comparison()
    fig_adversarial_cascade()
    fig_correlation_sensitivity()

    print()
    print('All figures generated successfully.')
    print(f'Files saved to: {OUT_DIR}')
