"""Plot motif enrichment using the saved unidentified notebook cluster IDs.

Reads printed counts rather than refitting KMeans, so the cluster numbers match
the notebook. These are motif-participation rows, not independent unique cells.
"""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MOTIF = 20748339660
ROOT = _REPO_ROOT


def saved_counts(notebook):
    """Read pandas' wrapped printed table, merging its column blocks by index."""
    doc = json.loads(notebook.read_text())
    marker = 'Distribution by cluster_15 (counts):'
    tables = []
    for cell in doc['cells']:
        for output in cell.get('outputs', []):
            text = ''.join(output.get('text', []))
            if marker not in text:
                continue
            text = text.split(marker, 1)[1].split('Distribution by cluster_15 (percentages', 1)[0]
            values, columns = {}, None
            for line in text.splitlines():
                if line.startswith('which_motif'):
                    columns = [int(x) for x in re.findall(r'\d+', line)]
                elif re.match(r'^\d+(?:\.0)?\s', line):
                    row = line.split()
                    cluster = int(float(row[0]))
                    counts = [int(x) for x in row[1:]]
                    if columns is None or len(columns) != len(counts):
                        raise ValueError('Unexpected printed count table format')
                    values.setdefault(cluster, {}).update(zip(columns, counts))
            table = pd.DataFrame.from_dict(values, orient='index').sort_index()
            if list(table.index) != list(range(15)) or table.isna().any().any():
                raise ValueError('Incomplete saved cluster table')
            tables.append(table.astype(int))
    if not tables or any(not table.equals(tables[0]) for table in tables[1:]):
        raise ValueError('Missing or inconsistent saved cluster counts')
    return tables[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=ROOT/'unidentified_post_process.ipynb')
    parser.add_argument('--output', type=Path, default=ROOT/'analysis/outputs/unidentified_motif_20748339660')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = saved_counts(args.notebook)
    result = pd.DataFrame({'motif_rows': counts[MOTIF], 'background_cells': counts[0]})
    result.index.name = 'cluster_15'
    result['motif_fraction'] = result.motif_rows / result.motif_rows.sum()
    result['background_fraction'] = result.background_cells / result.background_cells.sum()
    result['expected_motif_rows'] = result.background_fraction * result.motif_rows.sum()
    result['fold_enrichment'] = result.motif_fraction / result.background_fraction.replace(0, np.nan)
    result['percentage_point_difference'] = 100 * (result.motif_fraction - result.background_fraction)
    result.to_csv(args.output/'cluster_enrichment.csv')
    counts.rename_axis('cluster_15').to_csv(args.output/'notebook_saved_counts_all_motifs.csv')

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.15]})
    x = np.arange(15)
    axes[0].bar(x-.19, 100*result.background_fraction, .38, color='#9cabb6',
                label=f'Background: no selected motif (n={result.background_cells.sum():,})')
    axes[0].bar(x+.19, 100*result.motif_fraction, .38, color='#9672bb',
                label=f'This motif (n={result.motif_rows.sum():,} participation rows)')
    axes[0].set_ylabel('Share of population (%)')
    axes[0].legend(frameon=False, fontsize=9)
    colors = np.where(result.fold_enrichment > 1, '#b54c63', '#4f8fa8')
    axes[1].bar(x, result.fold_enrichment, color=colors, width=.72)
    axes[1].axhline(1, color='#333333', linestyle='--', linewidth=1)
    for i, row in result.iterrows():
        axes[1].text(i, row.fold_enrichment+.10, f'{row.fold_enrichment:.2f}×\nn={int(row.motif_rows)}',
                     ha='center', va='bottom', fontsize=8)
    axes[1].set_ylim(0, result.fold_enrichment.max()*1.2)
    axes[1].set_ylabel('Cluster enrichment (fold)')
    axes[1].set_xlabel('Unidentified-cell cluster (original notebook IDs)')
    axes[1].set_xticks(x)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Unidentified — Memory CD4 T — CD8 T — NK\nMotif 20748339660 · NN + NP · 15 protein-expression clusters', fontsize=13)
    fig.text(.5, .015, 'Enrichment = motif cluster share / background cluster share. Repeated motif participation is retained; descriptive only.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .96))
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(args.output/f'cluster_enrichment.{suffix}', dpi=200)
    plt.close(fig)
    (args.output/'README.md').write_text(
        '# Motif 20748339660: unidentified-cell cluster enrichment\n\n'
        'Path: Unidentified — Memory CD4 T cell — CD8 T cell — NK cell. '
        'Verified against `/groups/assafza_group/assafza/Omri/melanoma_motif_instances/20748339660.txt` '
        '(A=NK, B=CD8, C=Memory CD4, D=Unidentified; edges A–B, B–C, C–D).\n\n'
        'The figure uses the saved `cluster_15` count tables in `unidentified_post_process.ipynb`, '
        'preserving the original cluster IDs and NN/NP cohort. Background is `which_motif == 0` '
        '(unidentified cells outside all selected motifs), matching the notebook. '
        'Enrichment is (cluster motif rows / all motif rows) divided by '
        '(cluster background cells / all background cells). One-fold is the background distribution.\n\n'
        'The notebook joins motif hits before fitting KMeans, so a cell can appear repeatedly '
        'and influence both clustering and motif counts. The 644 motif rows are not 644 unique cells. '
        'The raw NN/NP hit file has 648 hits involving 485 unique unidentified cells before '
        'the notebook protein join. No significance claims or independent-cell confidence intervals '
        'are made. Cluster IDs differ from the separately recomputed unique-cell clusters in '
        '`unidentified_neighbor_enrichment`; do not transfer IDs between analyses.\n\n'
        'Reproduce: `MPLCONFIGDIR=/tmp/cism-mpl python analysis/marker_ownership/plot_unidentified_motif_cluster_enrichment.py`.\n'
    )
    print(result.sort_values('fold_enrichment', ascending=False).to_string())


if __name__ == '__main__':
    main()
