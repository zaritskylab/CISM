"""Fill Table S1 from existing historical FANMOD outputs; never run FANMOD."""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile


SOURCE = Path('/groups/assafza_group/assafza/Omri/processed_data_23_10_2025')
DATASET = 'melanoma_distance_100_XX'
NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
ET.register_namespace('x', NS)
ET.register_namespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
RECORD = re.compile(rb'^(\d+),([^\r\n]+)\r?$', re.MULTILINE)


def write_csv(path, records):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def read_labels():
    with (SOURCE / 'patient_class.csv').open() as handle:
        labels = {int(uid.removeprefix('Melanoma')): state for uid, state in csv.reader(handle)}
    return {pid: state for pid, state in labels.items() if state in ('NN', 'NP', 'PN')}


def audit_graphs(labels):
    graphs = {}
    for path in sorted((SOURCE / 'data' / DATASET).glob('Patient_*_FOV*.txt')):
        match = re.fullmatch(r'Patient_(\d+)_(FOV\d+)\.txt', path.name)
        assert match, path
        pid, fov = int(match[1]), match[2]
        if pid not in labels:
            continue
        raw = path.read_bytes()
        colors, nodes = set(), set()
        edge_rows = 0
        for line in raw.splitlines():
            if not line.strip():
                continue
            a, b, ca, cb = map(int, line.split())
            nodes.update((a, b))
            colors.update((ca, cb))
            edge_rows += 1
        # Historical notebook maps color 13 to Tumor (tutorial uses another mapping).
        assert 13 not in colors, f'Tumor color remains in {path}'
        graphs[pid, fov] = dict(graph_path=str(path), graph_sha256=hashlib.sha256(raw).hexdigest(),
                                graph_nodes=len(nodes), graph_node_id_slots=max(nodes) + 1,
                                graph_edge_rows=edge_rows,
                                tumor_color_13_present=False)
    assert Counter(labels.values()) == {'NN': 20, 'NP': 18, 'PN': 13}
    assert Counter(labels[pid] for pid, _ in graphs) == {'NN': 40, 'NP': 36, 'PN': 42}
    return graphs


def parse_output(path, pid, fov, state, size, graph):
    raw = path.read_bytes()
    header = raw[:4096].decode()
    assert f'Subgraph size: {size}' in header
    assert 'Generated 1000 random networks' in header
    assert 'Algorithm: enumeration' in header
    assert f'Patient_{pid}_{fov}.txt' in header
    # FANMOD allocates IDs 0..max_id, including gaps left by removed cells.
    assert int(re.search(r'Number of nodes: (\d+)', header)[1]) == graph['graph_node_id_slots']
    # FANMOD's edge header counts reciprocal pairs once for these undirected graphs.
    assert 2 * int(re.search(r'Number of edges: (\d+)', header)[1]) == graph['graph_edge_rows']
    enumerated = int(re.search(r'(\d+) subgraphs were enumerated in the original network', header)[1])
    identities = int(re.search(r'(\d+) motifs were found', header)[1])
    total = enriched = p_only = depleted = undefined_enriched = observed_ids = enriched_ids = 0
    rounded_count_rows = rounding_bound = enriched_rounding_bound = 0
    n_records = 0
    for match in RECORD.finditer(raw):
        fields = match[2].split(b',')
        assert len(fields) in (6, 7), (path, match[0][:150])
        freq, count, mean, std, z, p = fields[:6]
        numeric_count = float(count)
        assert numeric_count.is_integer(), (path, match[0])
        count = int(numeric_count)
        quantum = Decimal(fields[1].decode()).as_tuple().exponent
        count_bound = (10 ** quantum) / 2 if quantum > 0 else 0
        rounding_bound += count_bound
        rounded_count_rows += count_bound > 0
        freq, mean, p = float(freq), float(mean), float(p)
        assert count >= 0
        assert math.isfinite(p) and 0 <= p <= 1
        total += count
        n_records += 1
        observed_ids += count > 0
        if p < .05:
            p_only += count
            if freq > mean:
                # Undefined Z is allowed: e.g. positive observed frequency, null SD = 0.
                enriched += count
                enriched_rounding_bound += count_bound
                enriched_ids += count > 0
                if z == b'undefined':
                    undefined_enriched += count
                else:
                    assert float(z) > 0, (path, match[0])
            elif freq < mean:
                depleted += count
    # The header includes identities found only in random networks; the result
    # overview can omit these zero-observed-count identities.
    assert 0 < n_records <= identities, (path, n_records, identities)
    assert abs(total - enumerated) <= rounding_bound, (path, total, enumerated, rounding_bound)
    assert 0 <= enriched <= total
    return dict(patient_id=f'Melanoma{pid}', state=state, fov=fov, size=size,
                subgraph_instances=total, motif_instances=enriched,
                exact_subgraph_instances_header=enumerated,
                exported_sum_minus_header=total - enumerated,
                rounded_count_rows=rounded_count_rows,
                exported_subgraph_rounding_bound=rounding_bound,
                exported_motif_rounding_bound=enriched_rounding_bound,
                p_only_instances=p_only, depleted_significant_instances=depleted,
                enriched_undefined_z_instances=undefined_enriched,
                header_identities_found=identities,
                output_identity_rows=n_records, observed_identity_rows=observed_ids,
                enriched_observed_identity_rows=enriched_ids,
                random_networks=1000, raw_path=str(path), raw_sha256=hashlib.sha256(raw).hexdigest(),
                **graph)


def fill_workbook(template, output, patients, summary, legend):
    def set_cell(sheet, ref, value, formula=None):
        cell = sheet.find(f'.//{{{NS}}}c[@r="{ref}"]')
        assert cell is not None, ref
        for child in list(cell):
            cell.remove(child)
        cell.attrib.pop('t', None)
        if isinstance(value, str):
            cell.set('t', 'inlineStr')
            ET.SubElement(ET.SubElement(cell, f'{{{NS}}}is'), f'{{{NS}}}t').text = value
        else:
            if formula:
                ET.SubElement(cell, f'{{{NS}}}f').text = formula
            ET.SubElement(cell, f'{{{NS}}}v').text = str(value)

    with ZipFile(template) as original, ZipFile(output, 'w') as filled:
        for entry in original.infolist():
            raw = original.read(entry.filename)
            if entry.filename == 'xl/worksheets/sheet1.xml':
                sheet = ET.fromstring(raw)
                set_cell(sheet, 'A3', '51 patients: NN 20, NP 18, PN 13; 118 FOVs. Sums of exported counts; source rounding noted on second sheet.')
                for row, patient in enumerate(patients, 7):
                    for col, value in zip('ABCDEFGHI', patient.values()):
                        set_cell(sheet, f'{col}{row}', value)
                for col, key in zip('CDEFGHI', list(patients[0])[2:]):
                    set_cell(sheet, f'{col}58', sum(p[key] for p in patients), f'SUM({col}7:{col}57)')
                raw = ET.tostring(sheet, encoding='utf-8', xml_declaration=True)
            elif entry.filename == 'xl/worksheets/sheet2.xml':
                sheet = ET.fromstring(raw)
                updates = {
                    'A2': 'Data sources and checks for completed Table S1',
                    'B5': 'Completed: 51 patients (20 NN, 18 NP, 13 PN); PP excluded. IDs from historical patient_class.csv.',
                    'B6': 'Verified: 118 analyzed FOVs: NN 40, NP 36, PN 42. FOV sets agree across sizes 3, 4, 5.',
                    'B7': 'Sum exported Count across every identity and all FOVs. Some large Count values were rounded in the source (e.g. 1.4691e+05). Sums match exact header enumeration totals within the precision of exported counts; exact header totals and differences are in the per-FOV audit.',
                    'B8': 'Sum Count where FANMOD empirical p < 0.05 AND observed Frequency > Mean-Freq. Positive finite Z agrees; undefined Z is retained for enrichment (e.g. zero null SD). No additional Z cutoff or quantile filter.',
                    'B9': 'Historical processed_data_23_10_2025 dataset is configured as the precomputed PN-tumor-removed source in multi_class/config.py. All 42 PN graphs contain no Tumor color 13; graph node/edge totals match the output headers for all three sizes. Exact historical region-removal parameters are not independently recoverable from these outputs.',
                    'B10': str(SOURCE / 'fanmod_output' / DATASET) + '/{3,4,5}/*.csv; patient labels and input graphs from the same source root. No new FANMOD run.',
                    'B11': 'Occurrences summed across FOVs without deduplicating identities. Random-network-only identities contribute zero observed Count.',
                    'B12': 'Passed: 51 complete rows, 118 FOVs, 354 outputs; all motif counts <= subgraph counts. Overall reduction: ' + f"{summary['overall_reduction_percent']:.2f}%" + '. Source rounding affects ' + str(summary['outputs_with_rounded_counts']) + ' outputs. Manuscript claim was not available for comparison.',
                    'A14': 'SI legend (empirical one-tailed enrichment rule)',
                    'B15': legend,
                }
                for ref, value in updates.items():
                    set_cell(sheet, ref, value)
                    if ref.startswith('B'):
                        row = sheet.find(f'.//{{{NS}}}row[@r="{ref[1:]}"]')
                        row.set('ht', str(max(float(row.get('ht', 15)), math.ceil(len(value) / 90) * 15 + 12)))
                        row.set('customHeight', '1')
                raw = ET.tostring(sheet, encoding='utf-8', xml_declaration=True)
            filled.writestr(entry, raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = read_labels()
    graphs = audit_graphs(labels)
    records = []
    for size in (3, 4, 5):
        for i, ((pid, fov), graph) in enumerate(sorted(graphs.items()), 1):
            path = SOURCE / 'fanmod_output' / DATASET / str(size) / f'{pid}_{fov}.csv'
            records.append(parse_output(path, pid, fov, labels[pid], size, graph))
            if i % 20 == 0 or i == len(graphs):
                print(f'{size}-cell: verified {i}/{len(graphs)} FOV outputs', flush=True)
    write_csv(args.output_dir / 'Table_S1_melanoma_per_FOV_audit.csv', records)
    patients = []
    for state in ('NN', 'NP', 'PN'):
        for pid in sorted(pid for pid in labels if labels[pid] == state):
            patient = dict(patient_id=f'Melanoma{pid}', state=state,
                           FOVs=sum(p == pid for p, _ in graphs))
            for size in (3, 4, 5):
                selected = [r for r in records if r['patient_id'] == patient['patient_id'] and r['size'] == size]
                for metric in ('subgraph_instances', 'motif_instances'):
                    patient[f'{size}_cell_{metric}'] = sum(r[metric] for r in selected)
            patients.append(patient)
    write_csv(args.output_dir / 'Table_S1_melanoma_patient_counts.csv', patients)
    total = sum(r['subgraph_instances'] for r in records)
    enriched = sum(r['motif_instances'] for r in records)
    summary = dict(patients=len(patients), FOVs=len(graphs), outputs=len(records),
                   subgraph_instances=total, motif_instances=enriched,
                   overall_reduction_percent=100 * (1 - enriched / total),
                   outputs_with_rounded_counts=sum(r['rounded_count_rows'] > 0 for r in records),
                   exact_subgraph_instances_header=sum(r['exact_subgraph_instances_header'] for r in records),
                   exported_sum_minus_header=sum(r['exported_sum_minus_header'] for r in records),
                   exported_subgraph_rounding_bound=sum(r['exported_subgraph_rounding_bound'] for r in records),
                   exported_motif_rounding_bound=sum(r['exported_motif_rounding_bound'] for r in records),
                   motif_rule='FANMOD empirical p < 0.05 and Frequency > Mean-Freq; retain undefined Z when enriched',
                   by_size={str(size): {
                       metric: sum(r[metric] for r in records if r['size'] == size)
                       for metric in ('subgraph_instances', 'motif_instances', 'p_only_instances',
                                      'depleted_significant_instances', 'enriched_undefined_z_instances')
                   } for size in (3, 4, 5)})
    (args.output_dir / 'Table_S1_melanoma_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    legend = ('Table S1. Counts of connected subgraph instances and statistically enriched motif instances in 51 melanoma patients '
              '(20 NN, 18 NP, 13 PN; 118 FOVs). For each patient and subgraph size (3–5 cells), observed occurrences were summed '
              'across analyzed FOVs. Motif instances were restricted to identities with FANMOD+ empirical p < 0.05 and observed '
              'frequency greater than the mean frequency in 1,000 FOV-specific degree-preserving randomized graphs. '
              'Enriched identities with undefined Z scores were retained. PN FOVs use the historical tumor-removed input graphs. '
              'Counts refer to instances, not distinct identities; PP patients were excluded. '
              'Values are sums of exported Count fields; some large source counts were rounded in scientific notation, '
              'so affected sums have the precision of the original exports.')
    (args.output_dir / 'Table_S1_legend.txt').write_text(legend + '\n')
    fill_workbook(args.template, args.output_dir / 'Table_S1_melanoma_filled.xlsx', patients, summary, legend)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
