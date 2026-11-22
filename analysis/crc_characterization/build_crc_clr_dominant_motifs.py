#!/usr/bin/env python3
"""Create a CLR-only dominant motif count panel and standalone motif drawings."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

import argparse
import base64
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/crc-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from analysis.crc_characterization.plot_crc_motifs_vs_stringency import LegacyReader

ROOT = _REPO_ROOT
DEFAULT_COUNTS = Path('/home/omriavit/research/discriminative_motifs_counts_CRC_4.csv')
DEFAULT_ARTIFACT = Path('/home/omriavit/research/cism_CRC_4.pkl')
DEFAULT_LABELS = Path('/groups/assafza_group/assafza/Datasets/CRC/CRC_patient_class.csv')
DEFAULT_OUTPUT = ROOT / 'analysis/outputs/crc_clr_dominant_motifs'
CELL_TYPES = ['Bcell', 'CD3T', 'CD4T', 'CD8T', 'DC', 'Endothelial', 'Fibroblast',
              'Lymphatic', 'Macrophage', 'Neuron', 'Neutrophil', 'Plasma', 'SMV',
              'Stroma', 'Treg', 'Tumor', 'Unidentified']
DISPLAY = {'Bcell': 'B cell', 'CD3T': 'CD3 T cell', 'CD4T': 'CD4 T cell', 'CD8T': 'CD8 T cell'}
PALETTE = ['#079a27', '#dc16bc', '#dfc400']


def decode_graph(value):
    return pickle.loads(base64.b64decode(value))


def plot_motif(graph, motif_id, output_path):
    graph = nx.Graph(graph)
    types = {node: CELL_TYPES[int(attrs['type'])] for node, attrs in graph.nodes(data=True)}
    positions = nx.spring_layout(graph, seed=17, k=1.1, iterations=100)
    palette = plt.get_cmap('tab20')
    type_colors = {name: palette(i) for i, name in enumerate(CELL_TYPES)}
    fig, ax = plt.subplots(figsize=(5, 4.2))
    nx.draw_networkx_edges(graph, positions, ax=ax, edge_color='#646b75', width=3.2)
    nx.draw_networkx_nodes(graph, positions, ax=ax,
                           node_color=[type_colors[types[n]] for n in graph.nodes],
                           node_size=1750, edgecolors='white', linewidths=2)
    labels = {n: DISPLAY.get(types[n], types[n]) for n in graph.nodes}
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax, font_size=9, font_weight='medium')
    ax.set_title(f'CLR dominant motif {motif_id}\n4-node motif', fontsize=13, pad=12)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, transparent=True, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', type=Path, default=DEFAULT_COUNTS)
    parser.add_argument('--artifact', type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument('--labels', type=Path, default=DEFAULT_LABELS)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--n-motifs', type=int, choices=[2, 3], default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(args.counts)
    labels = pd.read_csv(args.labels)[['Patient', 'Group']].dropna()
    labels['Patient'] = labels.Patient.astype(int)
    clr_patients = sorted(labels.loc[labels.Group.astype(int) == 1, 'Patient'].tolist())
    clr_rows = counts[counts.patient_class == 'CLR'].copy()
    selected = (clr_rows.groupby('ID', as_index=False)
                .agg(total_CLR_count=('Count', 'sum'), clr_patients_with_motif=('Patient', 'nunique'))
                .sort_values(['total_CLR_count', 'clr_patients_with_motif', 'ID'],
                             ascending=[False, False, True]).head(args.n_motifs))
    if selected.empty:
        raise ValueError('No CLR discriminative motifs found in the motif-count source')
    motif_ids = selected.ID.astype(int).tolist()
    source_by_motif = clr_rows[clr_rows.ID.isin(motif_ids)].copy()
    source_by_motif['Patient'] = source_by_motif.Patient.astype(int)
    per_patient = (source_by_motif.pivot_table(index='Patient', columns='ID', values='Count',
                                               aggfunc='sum', fill_value=0)
                   .reindex(clr_patients, fill_value=0))
    per_patient.index.name = 'Patient'
    per_patient.columns.name = 'Motif ID'
    per_patient.to_csv(args.output_dir / 'clr_motif_counts_by_patient.csv')
    selected.to_csv(args.output_dir / 'selected_clr_motifs.csv', index=False)

    fig, axes = plt.subplots(args.n_motifs, 1, figsize=(10, 2.25 * args.n_motifs), sharex=True, squeeze=False)
    x = np.arange(len(clr_patients))
    for i, motif_id in enumerate(motif_ids):
        ax = axes[i, 0]
        ax.bar(x, per_patient[motif_id].to_numpy(), width=.78, color=PALETTE[i], edgecolor='none')
        ax.set_yscale('log')
        ax.set_ylim(.7, max(100, float(per_patient[motif_id].max()) * 1.6))
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(f'Motif {i + 1}: {motif_id}  |  CLR pooled count = {int(selected.set_index("ID").loc[motif_id, "total_CLR_count"]):,}',
                     loc='left', color=PALETTE[i], fontsize=10, pad=4)
        ax.set_yticks([1, 10, 100, 1000])
        ax.set_yticklabels(['1', '10', '100', '1,000'], fontsize=8)
        ax.grid(axis='y', which='major', color='#e5e5e5', linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#bbbbbb')
        ax.tick_params(axis='both', length=0, labelsize=8, pad=3)
    axes[-1, 0].set_xticks(x, [str(p) for p in clr_patients])
    axes[-1, 0].set_xlabel('CLR patient', fontsize=10)
    fig.text(.015, .985, 'E', fontsize=14, va='top')
    fig.tight_layout(rect=[.02, 0, 1, .98], h_pad=.7)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(args.output_dir / f'crc_clr_dominant_motifs_panel_E.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Standalone dominant-motif panel for figure assembly. Show every patient
    # with a nonzero count, grouped by the true patient class.  The motif was
    # selected from the CLR-discriminative set, but it can still occur in DII.
    dominant_id = motif_ids[0]
    dominant_counts = counts.loc[counts.ID.astype(int) == dominant_id,
                                 ['Patient', 'patient_class', 'Count']].copy()
    dominant_counts['Patient'] = dominant_counts.Patient.astype(int)
    dominant_counts = (dominant_counts.groupby(['Patient', 'patient_class'], as_index=False)['Count'].sum())
    dominant_counts = dominant_counts.loc[dominant_counts.Count > 0].copy()
    dominant_counts['cohort_order'] = dominant_counts.patient_class.map({'CLR': 0, 'DII': 1})
    dominant_counts = dominant_counts.sort_values(['cohort_order', 'Patient']).reset_index(drop=True)
    dominant_counts.drop(columns='cohort_order').to_csv(
        args.output_dir / f'crc_motif_{dominant_id}_nonzero_patient_counts_by_class.csv', index=False)

    # Add a small visual gap at the cohort boundary.
    n_clr = int((dominant_counts.patient_class == 'CLR').sum())
    standalone_x = np.arange(len(dominant_counts), dtype=float)
    if 0 < n_clr < len(dominant_counts):
        standalone_x[n_clr:] += .65
    bar_colors = dominant_counts.patient_class.map({'CLR': '#4C92CF', 'DII': '#C94F52'}).tolist()
    fig, ax = plt.subplots(figsize=(10, 3.25))
    ax.bar(standalone_x, dominant_counts.Count.to_numpy(), width=.78,
           color=bar_colors, edgecolor='none')
    ax.set_yscale('log')
    ax.set_ylim(.7, max(100, float(dominant_counts.Count.max()) * 1.6))
    ax.set_ylabel('Count', fontsize=11)
    ax.set_xticks(standalone_x, [str(patient) for patient in dominant_counts.Patient])
    ax.set_xlabel('Patient', fontsize=11)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(['1', '10', '100', '1,000'])
    ax.grid(axis='y', which='major', color='#e5e5e5', linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#bbbbbb')
    ax.tick_params(axis='both', length=0, labelsize=9, pad=4)
    if 0 < n_clr < len(dominant_counts):
        boundary = (standalone_x[n_clr - 1] + standalone_x[n_clr]) / 2
        ax.axvline(boundary, color='#6b7280', linestyle=(0, (4, 4)), linewidth=1)
    ax.legend(handles=[
        Line2D([0], [0], marker='s', linestyle='none', markersize=9,
               markerfacecolor='#4C92CF', markeredgecolor='none', label='CLR'),
        Line2D([0], [0], marker='s', linestyle='none', markersize=9,
               markerfacecolor='#C94F52', markeredgecolor='none', label='DII'),
    ], frameon=False, loc='upper left', ncol=2, handletextpad=.4, columnspacing=1.2)
    fig.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(args.output_dir / f'crc_CLR_motif_{dominant_id}_patient_counts.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

    with args.artifact.open('rb') as handle:
        artifact = LegacyReader(handle).load()
    data = artifact.motifs_dataset.drop_duplicates('ID').set_index('ID')
    motif_meta = []
    for i, motif_id in enumerate(motif_ids):
        graph = decode_graph(data.loc[motif_id, 'motif'])
        image_path = args.output_dir / f'motif_{i + 1}_{motif_id}.png'
        plot_motif(graph, motif_id, image_path)
        motif_meta.append({'rank': i + 1, 'ID': motif_id,
                           'CLR_total_Count': int(selected.set_index('ID').loc[motif_id, 'total_CLR_count']),
                           'CLR_patients_with_motif': int(selected.set_index('ID').loc[motif_id, 'clr_patients_with_motif']),
                           'graph_nodes': int(graph.number_of_nodes()),
                           'stored_directed_edges': int(graph.number_of_edges()),
                           'displayed_undirected_edges': int(nx.Graph(graph).number_of_edges()),
                           'cell_types': [DISPLAY.get(CELL_TYPES[int(attrs['type'])], CELL_TYPES[int(attrs['type'])])
                                          for _, attrs in graph.nodes(data=True)],
                           'image': image_path.name})
    (args.output_dir / 'motif_images.json').write_text(json.dumps({
        'selection': 'Top CLR discriminative motifs by pooled Count, from saved 4-node CRC motif analysis',
        'patient_scope': 'CLR only (17 patients)', 'stringency': .55,
        'motifs': motif_meta}, indent=2) + '\n')
    print(selected.to_string(index=False))
    print(f'Panel and motif drawings saved to {args.output_dir}')


if __name__ == '__main__':
    main()
