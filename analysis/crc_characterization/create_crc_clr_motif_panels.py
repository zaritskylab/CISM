#!/usr/bin/env python3
"""Export CLR top-three motif counts, diagrams and actual centroid-map instances."""

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
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/crc-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import networkx as nx
import numpy as np
import pandas as pd
from analysis.crc_characterization.plot_crc_motifs_vs_stringency import LegacyReader

ROOT = _REPO_ROOT
OUT = ROOT / 'analysis/outputs/crc_clr_motif_panels'
SOURCE = Path('/home/omriavit/research')
TYPES = ['Bcell', 'CD3T', 'CD4T', 'CD8T', 'DC', 'Endothelial', 'Fibroblast',
         'Lymphatic', 'Macrophage', 'Neuron', 'Neutrophil', 'Plasma', 'SMV',
         'Stroma', 'Treg', 'Tumor', 'Unidentified']
COLORS = ['#29a326', '#ca21bc', '#d5bd00']
ROMAN = ['I', 'II', 'III']
NAMES = {'Bcell': 'B cell', 'DC': 'DC'}
CELL_COLORS = {'Bcell': '#cd76ad', 'DC': '#8c80c6'}


def save(fig, stem, transparent=False):
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'{stem}.{ext}', dpi=300, bbox_inches='tight', transparent=transparent)
    plt.close(fig)


def enumerate_induced(graph, motif):
    """Connected typed backtracking, deduplicated by actual tissue node set."""
    pools = {m: {n for n in graph if graph.nodes[n]['type'] == motif.nodes[m]['type']} for m in motif}
    root = min(motif, key=lambda n: (len(pools[n]), -motif.degree[n]))
    order = [root]
    while len(order) < len(motif):
        frontier = [n for n in motif if n not in order and any(motif.has_edge(n, q) for q in order)]
        order.append(min(frontier, key=lambda n: (len(pools[n]), -sum(motif.has_edge(n, q) for q in order))))
    hits, chosen, used = {}, {}, set()

    def visit(k):
        if k == len(order):
            key = tuple(sorted(used))
            hits.setdefault(key, dict(chosen))
            return
        m = order[k]
        candidates = pools[m].copy()
        for previous, node in chosen.items():
            if motif.has_edge(m, previous):
                candidates.intersection_update(graph.neighbors(node))
        for node in sorted(candidates - used):
            if any(graph.has_edge(node, other) != motif.has_edge(m, previous)
                   for previous, other in chosen.items()):
                continue
            chosen[m] = node
            used.add(node)
            visit(k + 1)
            used.remove(node)
            del chosen[m]
    visit(0)
    return list(hits.values())


def draw_motif(ax, motif, rank, motif_id):
    pos = nx.planar_layout(motif)
    nx.draw_networkx_edges(motif, pos, ax=ax, edge_color=COLORS[rank], width=5)
    colors = [CELL_COLORS[TYPES[motif.nodes[n]['type']]] for n in motif]
    nx.draw_networkx_nodes(motif, pos, ax=ax, node_color=colors, node_size=2300,
                           edgecolors='#666666', linewidths=.7, alpha=.65)
    nx.draw_networkx_nodes(motif, pos, ax=ax, node_color=colors, node_size=1050,
                           edgecolors='none', alpha=.85)
    nx.draw_networkx_labels(motif, pos, ax=ax,
                           labels={n: NAMES[TYPES[motif.nodes[n]['type']]] for n in motif}, font_size=9)
    ax.set_title(f'{ROMAN[rank]}   ·   Motif {motif_id}', fontsize=11, pad=14)
    ax.margins(.35)
    ax.axis('off')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    artifact_path = SOURCE / 'cism_CRC_4.pkl'
    with artifact_path.open('rb') as handle:
        data = LegacyReader(handle).load().motifs_dataset.copy()
    selected = pd.read_csv(ROOT / 'analysis/outputs/crc_pairwise/selected_motifs.csv')
    labels = pd.read_csv(ROOT / 'analysis/outputs/crc_stringency/crc_patient_labels.csv').set_index('Patient').patient_class
    clr_patients = sorted(labels[labels == 'CLR'].index)
    clr = data[data.Patient.isin(clr_patients) & data.ID.isin(selected[selected.patient_class == 'CLR'].ID)]
    ranking = clr.groupby('ID', observed=True).agg(total_CLR_count=('Count', 'sum'),
                    CLR_patients_with_occurrences=('Patient', 'nunique')).sort_values('total_CLR_count', ascending=False)
    ranking = ranking.join(selected.set_index('ID').mean_patient_frequency)
    ranking.to_csv(OUT / 'CLR_motif_ranking.csv')
    ids = ranking.head(3).index.tolist()
    counts = clr[clr.ID.isin(ids)].groupby(['ID', 'Patient'], observed=True).Count.sum().unstack(fill_value=0)
    counts = counts.reindex(index=ids, columns=clr_patients, fill_value=0)
    counts.to_csv(OUT / 'panel_E_patient_counts.csv')
    templates = {}
    for motif_id in ids:
        obj = pickle.loads(base64.b64decode(data[data.ID == motif_id].iloc[0].motif))
        graph = nx.Graph(obj)
        for node in graph:
            graph.nodes[node]['type'] = int(graph.nodes[node]['type'])
        templates[motif_id] = graph
        nx.write_graphml(graph, OUT / f'motif_{motif_id}.graphml')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.7), sharex=True)
        for rank, motif_id in enumerate(ids):
            ax = axes[rank]
            ax.bar(np.arange(len(clr_patients)), counts.loc[motif_id], color=COLORS[rank], width=.75)
            ax.set_yscale('symlog', linthresh=1, linscale=.5)
            ax.set_ylim(0, counts.loc[motif_id].max() * 1.6)
            ticks = [0, 1, 10, 100, 1000, 10000]
            ax.set_yticks([t for t in ticks if t < ax.get_ylim()[1]])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.set_title(f'{ROMAN[rank]}   Motif {motif_id}', loc='left', fontsize=10, color=COLORS[rank], pad=4)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#cccccc')
            ax.tick_params(length=0, labelsize=8)
            ax.set_xticks(range(len(clr_patients)), clr_patients)
            ax.tick_params(axis='x', labelbottom=True)
        axes[0].text(-.095, 1.28, 'E', transform=axes[0].transAxes, fontsize=16)
        axes[0].text(.5, 1.26, 'CLR patients', transform=axes[0].transAxes,
                     ha='center', color='#2166ac', fontsize=12)
        fig.supylabel('Motif count', x=.015, fontsize=12)
        axes[-1].set_xlabel('Patient', fontsize=11)
        fig.tight_layout(rect=(.04, 0, 1, .97), h_pad=1)
        save(fig, 'panel_E_CLR_motif_counts')
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
        for rank, motif_id in enumerate(ids):
            draw_motif(axes[rank], templates[motif_id], rank, motif_id)
            single, ax = plt.subplots(figsize=(4, 3.7))
            draw_motif(ax, templates[motif_id], rank, motif_id)
            save(single, f'panel_D_motif_{ROMAN[rank]}_{motif_id}', transparent=True)
        fig.suptitle('CLR discriminative motifs', color='#2166ac', fontsize=14)
        save(fig, 'panel_D_CLR_motif_diagrams', transparent=True)

        # Pick the CLR patient with the highest pooled count of the three motifs.
        patient = int(counts.sum().idxmax())
        raw = pd.read_csv('/groups/assafza_group/assafza/Datasets/CRC/CRC_cells.csv',
                         usecols=['cellID', 'fov', 'centroid_x', 'centroid_y', 'class'])
        raw = raw[raw.fov.str.extract(r'^P(\d+)_')[0].astype(int) == patient]
        instance_rows, overlay_outputs, validations = [], [], []
        for path in sorted((SOURCE / 'CRC_graphs').glob(f'Patient_{patient}_FOV*.txt')):
            fov = int(path.stem.split('FOV')[1])
            cells = raw[raw.fov.str.endswith(f'_FOV{fov}')].reset_index(drop=True)
            if cells.empty:
                raise ValueError(f'No coordinates for {path.name}')
            graph = nx.Graph()
            for line in path.read_text().splitlines():
                u, v, a, b = map(int, line.split())
                graph.add_node(u, type=a); graph.add_node(v, type=b); graph.add_edge(u, v)
            for n, attrs in graph.nodes(data=True):
                assert TYPES[attrs['type']] == cells.iloc[n]['class'], 'Graph/coordinate node alignment failed'
            hits_by_id = {}
            for motif_id in ids:
                hits = enumerate_induced(graph, templates[motif_id])
                hits_by_id[motif_id] = hits
                recorded = int(data[(data.Patient == patient) & (data.FOV.astype(str) == f'FOV{fov}') & (data.ID == motif_id)].Count.sum())
                validations.append({'patient': patient, 'FOV': fov, 'ID': int(motif_id),
                                    'enumerated_instances': len(hits), 'FANMOD_count': recorded})
                for index, mapping in enumerate(hits):
                    for motif_node, graph_node in mapping.items():
                        cell = cells.iloc[graph_node]
                        instance_rows.append({'Patient': patient, 'FOV': fov, 'ID': int(motif_id),
                            'instance': index, 'motif_node': motif_node, 'graph_node': graph_node,
                            'cellID': int(cell.cellID), 'cell_type': cell['class'],
                            'x': cell.centroid_x, 'y': cell.centroid_y})
            pos = cells[['centroid_x', 'centroid_y']].to_numpy()

            def tissue_plot(stem, visible_ids, limits=None):
                fig, ax = plt.subplots(figsize=(7, 7), facecolor='black')
                ax.set_facecolor('black')
                ax.scatter(pos[:, 0], pos[:, 1], c='#cccccc', s=2, alpha=.65, linewidths=0)
                for rank, motif_id in enumerate(ids):
                    if motif_id not in visible_ids:
                        continue
                    edges, nodes = set(), set()
                    for mapping in hits_by_id[motif_id]:
                        nodes.update(mapping.values())
                        for u, v in templates[motif_id].edges():
                            edges.add(tuple(sorted([mapping[u], mapping[v]])))
                    segments = [[pos[u], pos[v]] for u, v in sorted(edges)]
                    ax.add_collection(LineCollection(segments, colors=COLORS[rank], linewidths=.8, alpha=.8))
                    if nodes:
                        pts = pos[sorted(nodes)]
                        ax.scatter(pts[:, 0], pts[:, 1], c=COLORS[rank], s=6, linewidths=0, alpha=.9)
                if limits:
                    ax.set_xlim(limits[0], limits[1]); ax.set_ylim(limits[2], limits[3])
                ax.invert_yaxis(); ax.set_aspect('equal'); ax.axis('off')
                ax.set_title(f'CLR patient {patient} · FOV {fov}', color='white', fontsize=12)
                handles = [Line2D([], [], color=COLORS[r], lw=3, label=f'{ROMAN[r]}: {len(hits_by_id[i])} instances')
                           for r, i in enumerate(ids) if i in visible_ids]
                legend = ax.legend(handles=handles, loc='upper left', frameon=True,
                                   facecolor='black', edgecolor='none', fontsize=8)
                for text in legend.get_texts():
                    text.set_color('white')
                save(fig, stem)
                overlay_outputs.append(stem)

            tissue_plot(f'panel_F_patient_{patient}_FOV{fov}_all_motifs', ids)
            # A crop chosen by motif-cell density, without altering the graph.
            involved = sorted({n for hits in hits_by_id.values() for mapping in hits for n in mapping.values()})
            if involved:
                points = pos[involved]
                span = .40 * min(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]))
                density = [np.sum((np.abs(points[:, 0] - p[0]) <= span / 2) &
                                  (np.abs(points[:, 1] - p[1]) <= span / 2)) for p in points]
                center = points[int(np.argmax(density))]
                limits = [center[0]-span/2, center[0]+span/2, center[1]-span/2, center[1]+span/2]
                tissue_plot(f'panel_F_patient_{patient}_FOV{fov}_crop', ids, limits)
                for rank, motif_id in enumerate(ids):
                    tissue_plot(f'panel_F_patient_{patient}_FOV{fov}_motif_{ROMAN[rank]}_crop', [motif_id], limits)
            # Export the full per-cell coordinates for assembling custom panels.
            cells.to_csv(OUT / f'patient_{patient}_FOV{fov}_cells.csv', index=False)
        pd.DataFrame(instance_rows).to_csv(OUT / 'panel_F_motif_instance_cells.csv', index=False)
        pd.DataFrame(validations).to_csv(OUT / 'instance_count_validation.csv', index=False)
    provenance = {'motifs': [int(i) for i in ids], 'ranking': 'Total Count across CLR patients, among selected CLR motifs at stringency 0.55',
                  'panel_E': 'All 17 CLR patients, numerical order; pooled FOV Count; zeros retained; symlog axis with linear 0–1 region',
                  'motif_color': dict(zip(ROMAN, COLORS)), 'overlay_patient': patient,
                  'overlays': 'Actual induced typed matches in saved undirected CRC graph; deduplicated by tissue node set. Centroid maps, not microscopy images.',
                  'graph_coordinate_alignment': 'Every graph node type verified against original per-FOV row index',
                  'instance_validation': validations, 'overlay_files': overlay_outputs,
                  'artifact': str(artifact_path), 'artifact_sha256': hashlib.sha256(artifact_path.read_bytes()).hexdigest()}
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(ranking.head(3).to_string())
    print('Representative CLR patient:', patient)
    print('Instance checks:', validations)
    print('Outputs:', OUT)


if __name__ == '__main__':
    main()
