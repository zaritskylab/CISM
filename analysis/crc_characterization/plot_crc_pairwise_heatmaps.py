#!/usr/bin/env python3
"""CRC motif-induced and full-graph unordered interaction probabilities."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT

import base64
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crc-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from analysis.crc_characterization.plot_crc_motifs_vs_stringency import LegacyReader

ROOT = _REPO_ROOT
OUT = ROOT / "analysis/outputs/crc_pairwise"
SOURCE = Path("/home/omriavit/research")
CELL_TYPES = ['Bcell', 'CD3T', 'CD4T', 'CD8T', 'DC', 'Endothelial',
              'Fibroblast', 'Lymphatic', 'Macrophage', 'Neuron', 'Neutrophil',
              'Plasma', 'SMV', 'Stroma', 'Treg', 'Tumor', 'Unidentified']
DISPLAY = {'Bcell': 'B cells', 'CD3T': 'CD3 T cells', 'CD4T': 'CD4 T cells', 'CD8T': 'CD8 T cells'}


def select_fold_union(data, threshold=0.55):
    significant = data[data.p_value < 0.05].copy()
    selected, folds = set(), []
    for patient in data.Patient_uId.unique():
        train = significant[significant.Patient_uId != patient].copy()
        denominator = train.groupby('patient_class', observed=True).Patient_uId.nunique()
        nclasses = train.groupby('ID', observed=True).patient_class.nunique()
        train = train[train.ID.isin(nclasses[nclasses == 1].index) & (train.nunique_colors >= 1)].copy()
        train['patient_count'] = train.groupby('ID', observed=True).Patient_uId.transform('nunique')
        train['prevalence'] = train.patient_count / train.patient_class.astype(str).map(denominator)
        # Preserve notebook behavior: cap ranked rows before taking unique IDs.
        chosen = train[train.prevalence > threshold].sort_values('Freq', ascending=False)
        ids = chosen.groupby('patient_class', observed=True).head(30).ID.unique()
        selected.update(ids)
        folds.extend({'held_out_patient': patient, 'ID': int(i)} for i in ids)
    return selected, pd.DataFrame(folds)


def normalize(matrix):
    assert np.isfinite(matrix).all() and (matrix >= 0).all() and matrix.sum() > 0
    result = matrix / matrix.sum()
    assert np.isclose(result.sum(), 1) and np.allclose(np.tril(result, -1), 0)
    return result


def plot_panel(ax, matrix, group, title, vmax):
    masked = np.ma.masked_where((matrix == 0) | np.tril(np.ones_like(matrix, dtype=bool), -1), matrix)
    cmap = plt.get_cmap('Blues' if group == 'CLR' else 'Reds').copy()
    cmap.set_bad('white')
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, interpolation='none')
    labels = [DISPLAY.get(n, n) for n in CELL_TYPES]
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, j in zip(*np.where(matrix > 0)):
        val = matrix[i, j]
        text = f'{val:.3f}' if val >= .0005 else '<.001'
        ax.text(j, i, text, ha='center', va='center', fontsize=4.2,
                color='white' if val / vmax > .57 else '#444444')
    ax.set_title(title, fontsize=10, pad=10)
    ax.text(.02, .10, group, transform=ax.transAxes, fontsize=12,
            color='#2166ac' if group == 'CLR' else '#b2182b')
    cb = ax.figure.colorbar(im, ax=ax, fraction=.035, pad=.025)
    cb.set_label('Probability', fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    artifact_path = SOURCE / 'cism_CRC_4.pkl'
    with artifact_path.open('rb') as handle:
        artifact = LegacyReader(handle).load()
    data = artifact.motifs_dataset.copy()
    labels_path = Path('/groups/assafza_group/assafza/Datasets/CRC/CRC_patient_class.csv')
    raw = pd.read_csv(labels_path)[['Patient', 'Group']].dropna()
    labels = raw.set_index(raw.Patient.astype(int)).Group.astype(int).map({1: 'CLR', 2: 'DII'})
    data['patient_class'] = pd.Categorical(data.Patient.map(labels))
    assert data.patient_class.notna().all()
    ids, folds = select_fold_union(data)
    saved_path = SOURCE / 'discriminative_motifs_counts_CRC_4.csv'
    assert ids == set(pd.read_csv(saved_path).ID), 'Selection differs from saved 119-motif result'
    folds.to_csv(OUT / 'selected_motifs_by_fold.csv', index=False)

    selected = data[data.ID.isin(ids)].copy()
    # Match GetMotifsResult: assign by majority of unique patient presence,
    # including nonsignificant occurrences. Resolve a tie in favor of CLR.
    presence = selected.groupby(['ID', 'patient_class'], observed=True).Patient_uId.nunique().unstack(fill_value=0)
    assignment = pd.Series(np.where(presence.CLR >= presence.DII, 'CLR', 'DII'), index=presence.index)
    selected['Total'] = selected.Count / selected.Freq
    patient_weight = selected.groupby(['Patient', 'ID'])[['Count', 'Total']].sum()
    weights = (patient_weight.Count / patient_weight.Total).groupby('ID').mean()
    motif_counts = {g: int((assignment == g).sum()) for g in ['CLR', 'DII']}
    summary_table = pd.DataFrame({'patient_class': assignment, 'mean_patient_frequency': weights})
    summary_table.to_csv(OUT / 'selected_motifs.csv')
    induced = {g: np.zeros((17, 17)) for g in ['CLR', 'DII']}
    representatives = selected.drop_duplicates('ID').set_index('ID')
    for motif_id, group in assignment.items():
        graph = pickle.loads(base64.b64decode(representatives.loc[motif_id, 'motif']))
        assert graph.number_of_nodes() == 4
        for u, v in nx.Graph(graph).edges():
            a, b = sorted([int(graph.nodes[u]['type']), int(graph.nodes[v]['type'])])
            induced[group][a, b] += weights.loc[motif_id]

    full = {g: np.zeros((17, 17)) for g in ['CLR', 'DII']}
    edge_totals, graph_patients, graph_hash = {g: 0 for g in full}, set(), hashlib.sha256()
    files = sorted((SOURCE / 'CRC_graphs').glob('Patient_*_FOV*.txt'))
    for path in files:
        graph_hash.update(path.name.encode()); graph_hash.update(path.read_bytes())
        patient = int(re.match(r'Patient_(\d+)_', path.name)[1])
        graph_patients.add(patient)
        group = labels.loc[patient]
        graph = nx.Graph()
        for line in path.read_text().splitlines():
            u, v, a, b = line.split()
            graph.add_node(u, cell_type=int(a)); graph.add_node(v, cell_type=int(b))
            graph.add_edge(u, v)
        edge_totals[group] += graph.number_of_edges()
        for u, v in graph.edges():
            a, b = sorted([graph.nodes[u]['cell_type'], graph.nodes[v]['cell_type']])
            full[group][a, b] += 1
    assert graph_patients == set(labels.index)
    for group in full:
        assert full[group].sum() == edge_totals[group]
    matrices = {'motif_induced': {g: normalize(m) for g, m in induced.items()},
                'all_pairwise': {g: normalize(m) for g, m in full.items()}}
    for kind, pair in matrices.items():
        for group, matrix in pair.items():
            pd.DataFrame(matrix, index=CELL_TYPES, columns=CELL_TYPES).to_csv(OUT / f'{kind}_{group}.csv')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for row, (kind, pair) in enumerate(matrices.items()):
            vmax = max(m.max() for m in pair.values())
            for col, group in enumerate(['CLR', 'DII']):
                title = (f'Discriminative motif-induced interactions (n={motif_counts[group]} motifs)'
                         if row == 0 else 'All graph pairwise interactions')
                plot_panel(axes[row, col], pair[group], group, title, vmax)
            axes[row, 0].text(-.32, 1.04, 'A' if row == 0 else 'B',
                              transform=axes[row, 0].transAxes, fontsize=14)
        fig.tight_layout(h_pad=3, w_pad=3)
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(OUT / f'crc_pairwise_heatmaps.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        for row, (kind, pair) in enumerate(matrices.items()):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            vmax = max(m.max() for m in pair.values())
            for col, group in enumerate(['CLR', 'DII']):
                plot_panel(axes[col], pair[group], group,
                           f'Motif-induced (n={motif_counts[group]} motifs), stringency 0.55' if row == 0 else 'All graph pairwise interactions', vmax)
            fig.tight_layout(w_pad=3)
            for ext in ['png', 'pdf', 'svg']:
                fig.savefig(OUT / f'crc_{kind}_heatmaps.{ext}', dpi=300, bbox_inches='tight')
            plt.close(fig)
    metadata = {'motif_size': 4, 'stringency': .55, 'recorded_ROC_AUC': .9199346405228759,
                'optimality_source': 'CISM_on_CRC.ipynb cell 24 recorded sweep; not a new tuning run',
                'selection': 'Union of leave-one-patient-out training selections; p<0.05; class-exclusive; prevalence>0.55; top 30 rows per class by Freq then unique IDs',
                'selected_total': len(ids), 'selected_by_class': motif_counts,
                'selection_validation': 'Exact ID-set equality with saved 119-motif CRC table',
                'class_assignment': 'Majority patient presence across all occurrences; ties to CLR',
                'motif_weight': 'Mean over observed patients of sum(Count)/sum(Count/Freq) across FOVs, matching GetMotifsResult',
                'pair_definition': 'Unordered cell-type pairs; one count per undirected edge; all probability mass in upper triangle including diagonal',
                'full_graph_edges': edge_totals, 'graph_files': len(files),
                'cohort': labels.value_counts().to_dict(),
                'color_scales': 'Zero-based, shared across CLR and DII within each row',
                'artifact': str(artifact_path), 'artifact_sha256': hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                'graph_directory': str(SOURCE / 'CRC_graphs'), 'graph_sha256': graph_hash.hexdigest(),
                'labels': str(labels_path)}
    (OUT / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (OUT / 'README.md').write_text(
        '# CRC pairwise interactions\n\n'
        '(A) Discriminative 4-node motif-induced interactions at the notebook’s best recorded '
        'stringency (0.55; ROC AUC 0.920). The 119-motif fold union exactly matches the saved CRC '
        'selection. Motifs are assigned by majority patient presence and weighted by mean patient '
        'frequency, as in the notebook. (B) All graph interactions pooled across patients/FOVs. '
        'CLR is blue; DII is red. Each panel is a distribution over unordered cell-type pairs, '
        'including same-type interactions, and sums to one. Each undirected edge contributes once; '
        'lower-triangle values are folded into the upper triangle, never discarded. Shared zero-based '
        'color scales within each row allow magnitude comparison. Labels below 0.0005 are shown '
        'as <.001; CSVs retain full precision.\n\n'
        'Reproduce: `python analysis/crc_characterization/plot_crc_pairwise_heatmaps.py`. '
        'These are motif-template-weighted interactions, following the existing analysis, rather '
        'than newly enumerated physical motif occurrences. Sources and settings: `provenance.json`.\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
