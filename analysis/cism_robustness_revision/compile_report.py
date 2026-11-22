#!/usr/bin/env python3
"""Compile existing numerical results into one analysis per page; no new fits."""
from pathlib import Path
import json
import html
import math
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cism-robustness-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'compiled_report'
DEST.mkdir(exist_ok=True)
PAGES = []

def read(name): return pd.read_csv(ROOT / name)
def js(name): return json.loads((ROOT / name).read_text())
def f(x, digits=4): return 'Undefined' if pd.isna(x) else f'{x:.{digits}f}'
def interval(lo, hi): return f'{f(lo)}–{f(hi)}'
def page(title, figures, cohort):
    p = dict(title=title, figures=figures, blocks=[])
    PAGES.append(p)
    paragraph(p, cohort)
    return p

def paragraph(p, text): p['blocks'].append(dict(type='paragraph', text=text))
def heading(p, text): p['blocks'].append(dict(type='heading', text=text))
def table(p, headers, rows, widths=None):
    assert all(len(row)==len(headers) for row in rows)
    p['blocks'].append(dict(type='table', headers=headers, rows=[[str(c) for c in row] for row in rows], widths=widths))

def uncertainty_note(p,n_runs=30):
    paragraph(p, f'SD is sample SD across {n_runs} classifier seeds (0–{n_runs-1}), not a standard error. The 95% CI is a conditional percentile interval from 10,000 stratified patient resamples of the seed-0 held-out predictions. It belongs to seed 0, not to the {n_runs}-seed mean. The bootstrap does not refit models or adjust for parameter selection.')

def sources(p, names): paragraph(p, 'Numerical sources (relative to analysis/cism_robustness_revision/): ' + '; '.join(names) + '.')

primary = read('primary_classifier_summary.csv').set_index('name')
binary = read('additional/binary_summary.csv').set_index('name')
mc = read('additional/multiclass_summary.csv')
mc_runs=int(mc.n_runs.iloc[0])
assert set(mc.n_runs)=={mc_runs}
H = ['Representation / source', 'Mean AUC ± SD', 'AUC range', 'Seed-0 AUC', 'Seed-0 95% CI']
W = [2.2, 1.5, 1.5, 1, 1.6]
def prow(name, label):
    r=primary.loc[name]
    return [label, f'{f(r.mean_auc)} ± {f(r.sd_auc)}', interval(r.min_auc,r.max_auc), f(r.reference_auc), interval(r.reference_ci_low,r.reference_ci_high)]
def brow(name, label):
    r=binary.loc[name]
    return [label, f'{f(r.mean_auc)} ± {f(r.sd_auc)}', interval(r.min_auc,r.max_auc), f(r.reference_auc), interval(r.ci_low,r.ci_high)]

p=page('Melanoma NN–NP classification', 'Fig. 2B–D; Figs. S4–S6; related robustness: Figs. S8–S9 and SStability', '38 patients: 20 NN and 18 NP; 76 FOVs. Patient-level LOOCV; NN is the positive class.')
table(p,H,[prow('nn_np_size4','CISM: original multi-size tables'),brow('melanoma_nn_np_cell_type','Cell-type composition'),brow('melanoma_nn_np_pairwise','Historical pairwise features'),prow('nn_np_raw4','CISM: saved FANMOD run 0')],W)
heading(p,'Configuration and feature counts')
paragraph(p,'CISM: four-cell motifs; stringency 0.46; 19 fold-union motifs for each listed source; 100-tree Random Forest. Motifs are reselected inside each LOOCV training fold. Both fixed-table CISM analyses use 30 classifier seeds, with the execution seed reset in each fold.')
paragraph(p,'Cell type: 16 frequency features from retained graph nodes pooled across each patient’s FOVs. Pairwise: 120 distinct-type combinations; homotypic edges are excluded as predictors but retained in the denominator. Cell-type seeds reset each fold; pairwise RNG advances across fixed lexical patient-order folds.')
uncertainty_note(p)
heading(p,'Source distinction')
paragraph(p,'Original multi-size tables and saved FANMOD run 0 are different motif-significance realizations. Their results must remain separate. The 100 independent FANMOD executions and motif recovery are reported on dedicated pages. The manuscript’s historical AUC 0.88 should not silently receive the new 0.8903 prediction vector’s CI.')
paragraph(p,'Classification supplies auxiliary validation of spatial motif discovery. No GNN error estimates were computed in this work.')
sources(p,['primary_classifier_summary.csv','additional/binary_summary.csv','nn_np_size4_native_selection.json','additional/prepared/*_manifest.json'])

p=page('Melanoma PN–NP classification', 'Fig. 7 (cohort and motif interpretation); Fig. S12A–C (composition, stringency and classification)', '31 patients: 18 NP and 13 PN. Stored internally as NP–PN; NP is the positive class. Patient-level LOOCV.')
table(p,H,[prow('np_pn_size5_original_selector','CISM: original selector'),brow('melanoma_np_pn_cell_type','Cell-type composition'),brow('melanoma_np_pn_pairwise','Historical pairwise features')],W)
paragraph(p,'CISM: five-cell motifs; stringency 0.40; 210 fold-union motifs; all 31 training folds retain features. Random Forest: 100 trees, 30 seeds, integer seed reset in every fold. Native frequency ordering and the top-30-FOV-row cap are preserved.')
paragraph(p,'Cell type: 16 graph-node frequency features. Pairwise: 120 distinct-type combinations, with the historical homotypic-edge omission and denominator. Baseline cohorts and held-out units match the 31 patients above; cell-type seeds reset per fold and pairwise RNG advances across folds.')
uncertainty_note(p)
heading(p,'Implementation sensitivity: retain separately')
r=read('classifier_seed_summary.csv').set_index('name').loc['np_pn_size5']
table(p,['Selector','Motifs','Mean AUC ± SD','Range','Seed 0 / 95% CI'],[['Stable frequency ties',205,f'{f(r.mean_auc)} ± {f(r.sd_auc)}',interval(r.min_auc,r.max_auc),f'{f(r.reference_auc)} / {interval(r.reference_ci_low,r.reference_ci_high)}']],[1.7,.6,1.5,1.4,2.6])
paragraph(p,'The stable-tie first pass changes the selected sets in 15/31 folds and is not the primary Fig. 7 analysis. Native selection recovers the historical AUC 0.818376 and 210 motifs; the historical full prediction vector was unavailable for equality testing. Below-chance baseline results are retained without post hoc probability inversion.')
sources(p,['primary_classifier_summary.csv','additional/binary_summary.csv','classifier_seed_summary.csv','native_selection_parity.csv'])

p=page('TNBC survival classification', 'Current Fig. 8A (formerly Fig. 7A); Fig. 8C (33 motifs); Figs. S14–S16', '37 patients and 38 FOVs: 30 long-term survivors (≥1,000 days) and 7 short-term survivors (<1,000 days). Long-term survival is positive. The two FOVs from TNBC15 share one patient-level held-out prediction.')
t=js('tnbc/summary.json')['saved']
saved=['CISM: exact saved threshold',f'{f(t["mean_auc"])} ± {f(t["sd_auc"])}',interval(t['min_auc'],t['max_auc']),f(t['reference']['auc']),interval(t['reference']['ci_low'],t['reference']['ci_high'])]
table(p,H,[prow('tnbc_size3_fig8a_stringency_045','CISM: figure stringency 0.45'),brow('tnbc_cell_type','Cell-type composition'),brow('tnbc_pairwise','Historical pairwise features'),saved],W)
heading(p,'Two distinct CISM settings')
table(p,['Setting','Motif size','Stringency','Fold-union motifs'],[['Fig. 8A/C',3,'0.45',33],['Saved artifact',3,'0.4404145064107466',37]],[1.4,1,2.4,1.5])
paragraph(p,'Both settings use the same 30 seeds with paired per-fold RNG states from the historical evaluator’s third trial. Earlier trial forest seeds are reserved analytically. At the exact artifact setting, all 37 saved probabilities replay within 1.11 × 10⁻¹⁶. The literal 0.45 result is a controlled evaluation, not a replay of the complete original Fig. 8A RNG history.')
paragraph(p,'Cell type: 16 retained graph-node frequency features. Pairwise: 120 distinct-type combinations; homotypic edges are omitted as predictors but retained in normalization. All models use 100-tree forests and patient-level LOOCV. Native CISM feature selection occurs within training folds.')
uncertainty_note(p)
paragraph(p,'The 33-motif figure setting and 37-motif artifact setting are not interchangeable. Four-cell TNBC, full-curve uncertainty bands, repeated FANMOD executions, and GNN repetitions were not evaluated.')
sources(p,['tnbc/summary.json','tnbc/source_manifest.json','additional/binary_summary.csv'])

p=page('CRC CLR–DII classification', 'Fig. SCRC-Cohort (classification; provisional manuscript label); related Fig. SCRC-Permutation', '35 patients: 17 CLR and 18 DII. CLR is positive. Patient-level LOOCV.')
table(p,H,[prow('crc_size4_historical_rng','CISM: four-cell motifs'),brow('crc_cell_type','Raw-cell composition'),brow('crc_pairwise','Historical pairwise features')],W)
paragraph(p,'CISM: stringency 0.55; 119 fold-union motifs; all 35 training folds retain features. Thirty seeds vary the historical RandomState progression over the threshold grid, using 100 trees per forest. Seed 0 exactly reproduces all 35 saved CISM probabilities. The cell-type seed-0 probabilities also replay exactly.')
paragraph(p,'Cell type: 17 raw-cell frequency features, using the existing CRC singleton-classifier export. Pairwise: 136 distinct-type combinations from graphs; homotypic edges remain in the denominator but are omitted as predictors.')
paragraph(p,'CRC pairwise correction: the earlier reported AUC 0.5471 was invalid because feature rows and patient labels had different orders. The corrected row above uses aligned patient IDs; all 35 feature rows exactly match native graph-reader reconstruction. The saved CRC notebook independently reports pairwise mean AUC 0.9017974 and population SD 0.0114251 over 10 trials. Those historical trials and the new 30-seed fixed-order runs are separate estimates.')
uncertainty_note(p)
heading(p,'Historical stringency sweep: one execution')
table(p,['Stringency','AUC (rounded historical value)'],[[s,a] for s,a in [('0.40','0.866'),('0.45','0.877'),('0.50','0.882'),('0.55','0.920'),('0.60','0.881'),('0.70','0.863')]],[1,2])
paragraph(p,'Mean features per training fold: 41.6 at 0.40; 31.5 at 0.55; 5.2 at 0.70; 1.0 at 0.80. The sweep is descriptive, not repeated-run parameter uncertainty.')
paragraph(p,'CRC cell composition is strongly predictive (mean AUC 0.9180 versus CISM 0.9071). These estimates do not establish universal CISM classification superiority or a formal paired difference. Repeated CRC FANMOD executions and GNN repetitions remain unmeasured.')
sources(p,['primary_classifier_summary.csv','crc/summary.json','additional/binary_summary.csv','RESULTS.md (historical rounded sweep)'])

p=page('Multiclass melanoma classification', 'Multiclass performance panel from the referenced multiclass analysis; final manuscript figure number not verified', '51 patients: 20 NN, 18 NP, 13 PN. Final three-confidence-score stack: pairwise binary Random Forest confidence scores followed by L2 multinomial logistic regression under LOOCV.')
labels={'accuracy':'Accuracy','balanced_accuracy':'Balanced accuracy','roc_auc_ovr_macro':'Macro OvR AUC','roc_auc_ovr_NN':'NN OvR AUC','roc_auc_ovr_NP':'NP OvR AUC','roc_auc_ovr_PN':'PN OvR AUC'}
table(p,['Metric','Mean ± SD','Range','Seed-0 result','Seed-0 95% CI'],[[labels[r.metric],f'{f(r.mean)} ± {f(r.sd)}',interval(r.min,r.max),f(r.reference),interval(r.ci_low,r.ci_high)] for r in mc.itertuples()],[1.8,1.5,1.5,1.2,1.6])
paragraph(p,f'{mc_runs} seeds retrain all 105 binary scoring forests per execution and all 51 held-out meta-classifiers. Base forests have 100 trees. Logistic regression: L2, C=1, lbfgs, max_iter=2,000. Fixed binary stringencies: NN–NP 0.45; NN–PN 0.55; NP–PN 0.45.')
paragraph(p,'Reference replay: maximum confidence-matrix difference 2.22 × 10⁻¹⁶; maximum final-probability difference 8.28 × 10⁻¹⁴. The original seed-0 reference is retained separately from the repeated-run mean.')
if mc_runs==60:
    batches=read('additional/multiclass_batch_summary.csv')
    heading(p,'Original and additional trials: mean ± sample SD')
    rows=[]
    for batch,label in [('first_30','Original 30 (0–29)'),('additional_30','Additional 30 (30–59)'),('all_trials','All 60 (0–59)')]:
        group=batches[batches.batch==batch].set_index('metric')
        rows.append([label]+[f'{f(group.loc[k,"mean"])} ± {f(group.loc[k,"sd"])}' for k in ['accuracy','balanced_accuracy','roc_auc_ovr_macro']])
    table(p,['Batch (seeds)','Accuracy','Balanced accuracy','Macro OvR AUC'],rows,[1.8,1.5,1.5,1.5])
    counts=batches[batches.batch=='all_trials'].set_index('metric').n_above_reference
    paragraph(p,f'Executions exceeding the original reference (of 60): accuracy {counts["accuracy"]}; balanced accuracy {counts["balanced_accuracy"]}; macro AUC {counts["roc_auc_ovr_macro"]}. All trials enter the combined summary; none are selected for higher performance.')
table(p,['Chance reference','Value'],[['Majority-class accuracy (20/51)',f(20/51)],['Balanced accuracy (1/3)',f(1/3)],['OvR AUC',f(.5)]],[2,1])
uncertainty_note(p,mc_runs)
paragraph(p,'Evaluation limitation: this preserves the original stack, which is not fully nested. Training confidence features are reused across outer meta folds, and stringencies were chosen using this cohort. Neither seed SD nor these conditional CIs corrects for those dependencies or establishes unbiased post-selection generalization.')
sources(p,['additional/multiclass_summary.csv','additional/multiclass_batch_summary.csv','additional/reference_replays.json','additional/prepared/multiclass_confidence_stack_manifest.json'])

p=page('Underrepresented spatial motifs', 'Fig. SUnderrepresented (four-cell, two-tailed normal-Z NN–NP analysis)', '38 patients: 20 NN and 18 NP. NN is positive. Four-cell underrepresented motifs, fixed stringency 0.35; 30 classifier seeds.')
table(p,H,[brow('underrepresented_4node_035','Four-cell anti-motifs')],W)
table(p,['Quantity','Value'],[['Fold-union motifs','60'],['Training folds retaining features','38/38'],['Historical AUC at 0.35','0.6958333333333333'],['New seed-0 AUC','0.6958333333333333'],['Bootstrap replicates per interval','10,000']],[2,2.5])
paragraph(p,'Features are positive depletion magnitudes for observed motifs with two-tailed normal-Z p<0.05 and Z<0. Native CISM training-fold selection retains strict prevalence >0.35 and the top-30-FOV-row cap per class before motif-ID deduplication. Forests have 100 trees; each execution uses a deterministic advancing RNG with the original within-class training-patient ordering.')
heading(p,'Historical four-point curve: one execution')
table(p,['Stringency','Two-tailed four-cell AUC'],[[f(r.stringency,2),f(r.roc_auc)] for r in pd.read_csv(ROOT.parents[1]/'melanoma_tutorials/melanoma_tutorials/runtime/absent_discriminative_cism_nn_np/figures/auc_fixed_stringencies_two_vs_lower_tail.csv').itertuples() if r.test=='two_tailed'],[1,2])
uncertainty_note(p)
paragraph(p,'The historical threshold CSV stores AUC only. The AUC matches exactly, but equality of historical and new patient probabilities cannot be established. The CI uses the new saved seed-0 predictions. Historical fitting used a shared threaded RNG; the new repeats use sequential deterministic RNG progression. The lower-tail and three-cell variants are separate analyses and are not substituted here.')
sources(p,['additional/binary_summary.csv','additional/reference_replays.json','additional/prepared/underrepresented_4node_035_manifest.json'])

p=page('FANMOD execution and stringency robustness', 'Fig. SStability; Fig. 2D (related selected setting)', 'Melanoma NN–NP: 38 patients, 76 FOVs; 100 independently generated FANMOD table sets. Four-cell motifs. RF seed fixed at 0; no pooling with classifier-seed experiments.')
hist=js('historical_summary.json')
table(p,['Protocol','Mean AUC ± SD','Range','Mean motifs ± SD'],[[label,f'{f(hist[key]["mean_auc"])} ± {f(hist[key]["sd_auc"])}',interval(hist[key]['min_auc'],hist[key]['max_auc']),f'{hist[key]["mean_motifs"]:.2f} ± {hist[key]["sd_motifs"]:.2f}'] for key,label in [('fixed','Fixed stringency 0.46'),('optimal','Execution-selected optimum')]], [2,1.6,1.5,1.5])
paragraph(p,'Fixed-setting run 0: AUC 0.9208; conditional patient-bootstrap 95% CI 0.7986–1.0000; 19 motifs. The execution-selected optimum is descriptive and reuses the cohort for selection; it is not an independent validation estimate.')
heading(p,'Stringency sweep across the same 100 FANMOD executions')
d=read('historical_parameter_summary.csv')
table(p,['Stringency','Mean AUC ± SD','Valid / 100','Mean union motifs','Mean empty folds'],[[f(r.stringency,2),f'{f(r.mean_auc)} ± {f(r.sd_auc)}' if r.n_valid else 'Undefined',int(r.n_valid),f(r.mean_motifs,2),f(r.mean_empty_folds,1)] for r in d.itertuples()],[1,1.8,1,1.2,1.2])
paragraph(p,'AUC is undefined when any training fold has no selected features. At 0.60, 11/38 folds are empty in every execution; at tested settings ≥0.70 all 38 folds are empty. Stringencies 0.45 and 0.46 select the same features because prevalence is discrete; they are not independent evidence.')
paragraph(p,'FANMOD variation and RF variation are distinct sources. No joint FANMOD-plus-RF variance was estimated. CIs are conditional on fixed held-out predictions, without retraining or selection correction. All 100 source fingerprints and saved-prediction AUC recalculations passed audit.')
sources(p,['historical_summary.json','historical_parameter_summary.csv','historical_run_metrics.csv','historical_source_audit.csv'])

p=page('Discriminative motif recovery', 'Fig. SStability; reference motifs relate to Figs. S4–S5', 'Melanoma NN–NP, four-cell motifs, fixed stringency 0.46, 100 independent FANMOD executions. Recovery is the fraction of executions selecting a motif; it is not classifier uncertainty.')
recovery=read('motif_recovery.csv')
mapping=read('tutorial_reference_mapping.csv').set_index('raw_motif_id')
table(p,['Raw-library motif ID','Run-0 reference','Tutorial reference','Executions / 100','Recovery (%)'],[[int(r.motif_id),'Yes' if r.in_reference else 'No','Yes' if r.motif_id in mapping.index else 'No',int(r.selected_runs),int(r.recovery_percent)] for r in recovery.itertuples()],[1.7,1.2,1.3,1.4,1.2])
paragraph(p,'Across executions: 24 distinct identities selected at least once. Saved run-0 reference: 19 motifs, 18/19 recovered in >90% of executions; one recovered in 45%. Its five additional motifs peak at 41% recovery.')
paragraph(p,'Tutorial reference: 19 motifs (8 NN, 11 NP); all 19 match the repeated-run library by labelled graph isomorphism. Eighteen are recovered in >90% of runs and one in 41%. Relative to this reference, the raw run-0 motif recovered in 45% is additional. Numeric IDs alone are not comparable across encodings.')
paragraph(p,'The SI’s current claim of 17/19 above 90% plus a consistently recovered additional motif is not supported by either audited reference. The tutorial list’s exact identity with the S4/S5 artwork has not been independently established; choose the intended reference before changing the SI.')
sources(p,['motif_recovery.csv','tutorial_reference_mapping.csv','tutorial_reference_mapping_summary.json'])

# Primary native sweeps replace only the relevant motif-size column. Other
# sizes retain the documented stable-frequency-tie first-pass implementation.
sweeps=read('new_stringency_results.csv')
thresholds=[.2,.3,.4,.42,.44,.45,.46,.48,.5,.6,.7,.8]
for contrast,label,n,native_size,figures in [('nn_np','Melanoma NN–NP',38,4,'Fig. 2C–D; Fig. S6'),('np_pn','Melanoma PN–NP',31,5,'Fig. S12B–C; Fig. 7 motif panel')]:
    p=page(f'{label} parameter sensitivity',figures,f'{n} patients. Single RF-seed-0 LOOCV sweeps on fixed original multi-size caches. Figures motivate these descriptive settings; no patient-bootstrap or execution uncertainty was calculated at every grid point.')
    rows=[]
    for s in thresholds:
        row=[f(s,2)]
        for size in [3,4,5]:
            data=read(f'{contrast}_size{size}_native_sweep.csv') if size==native_size else sweeps[(sweeps.name==f'{contrast}_size{size}') & (sweeps.seed==0)]
            r=data[np.isclose(data.stringency,s)].iloc[0]
            row += [f(r.auc) if r.rf_admissible else '—',str(int(r.n_motifs)),str(int(r.empty_folds))]
        rows.append(row)
    table(p,['Stringency','3-cell AUC','3-cell M','3-cell E','4-cell AUC','4-cell M','4-cell E','5-cell AUC','5-cell M','5-cell E'],rows,[1.1,1.1,.8,.8,1.1,.8,.8,1.1,.8,.8])
    paragraph(p,'M = fold-union motif count; E = number of empty-feature training folds. A dash denotes undefined RF AUC because at least one fold has no features. Below-chance AUCs are retained without inversion.')
    paragraph(p,f'The {native_size}-cell column uses the original CISM frequency ordering. Other sizes use the documented stable-tie first-pass selector. These columns are not all an exact replay of historical plot values. Feature selection occurs within each training fold; the cap selects the top 30 FOV rows per class before motif-ID deduplication.')
    # Record the first-pass primary-size values too, to avoid hiding the
    # implementation sensitivity already measured in the initial task.
    heading(p,f'Frequency-tie implementation sensitivity: {native_size}-cell motifs')
    alt=sweeps[(sweeps.name==f'{contrast}_size{native_size}') & (sweeps.seed==0)]
    altrows=[]
    native=read(f'{contrast}_size{native_size}_native_sweep.csv')
    for s in thresholds:
        a=alt[np.isclose(alt.stringency,s)].iloc[0]
        b=native[np.isclose(native.stringency,s)].iloc[0]
        if (not np.isclose(a.auc,b.auc,equal_nan=True)) or a.n_motifs!=b.n_motifs:
            altrows.append([f(s,2),f(a.auc) if a.rf_admissible else '—',int(a.n_motifs),int(a.empty_folds)])
    if altrows:table(p,['Stringency','Stable-tie AUC','Stable-tie motifs','Empty folds'],altrows,[1,1.5,1.5,1])
    else:paragraph(p,'No AUC or fold-union motif-count differences at the 12 shared grid points.')
    paragraph(p,'Increasing stringency reduces discovery material and eventually prevents fitting. These sweeps are not nested model-selection validation and do not justify a claim of uniformly high performance across arbitrary parameter choices.')
    sources(p,[f'{contrast}_size{native_size}_native_sweep.csv','new_stringency_results.csv'])

pairs=read('pair_distribution_sensitivity.csv')
for contrast,label,classes in [('nn_np','Melanoma NN–NP','NN / NP'),('np_pn','Melanoma PN–NP','NP / PN')]:
    p=page(f'{label} motif-induced interaction robustness','Figs. S8–S9 (NN–NP); related NP–PN parameter analysis: Fig. S12B–C',f'Occurrence-weighted undirected motif-induced cell-type pair distributions; fold-union motifs. Class order: {classes}. Spearman coefficients are descriptive summaries of dependent pair-frequency entries.')
    d=pairs[(pairs.comparison==contrast)&(pairs.kind=='stringency')]
    table(p,['Motif size','Stringency pair','Class','Spearman ρ','Descriptive P','Cross-class ρ at right'],[[int(r.motif_size),f'{r.left:.2f} → {r.right:.2f}',r.label,f(r.spearman_rho),f'{r.descriptive_p:.3g}',f(r.cross_class_rho_at_right)] for r in d.itertuples()],[.9,1.5,.7,1.2,1.3,1.5])
    heading(p,'Motif-size comparison at stringency 0.30')
    d=pairs[(pairs.comparison==contrast)&(pairs.kind=='motif_size')]
    table(p,['Sizes (cells)','Class','Spearman ρ','Descriptive P'],[[f'{int(r.left)} → {int(r.right)}',r.label,f(r.spearman_rho),f'{r.descriptive_p:.3g}'] for r in d.itertuples()],[1.5,1,1.5,1.5])
    paragraph(p,'These are recomputed undirected, occurrence-weighted distributions, not an exact reconstruction of the original S8/S9 matrices or orientation convention. Feature selection uses the documented stable-frequency-tie first-pass convention. No patient-level CIs or independent biological replicate P values are available for these correlations.')
    paragraph(p,'Four- versus five-cell consistency is stronger than three- versus four-cell consistency. Some within-class threshold correlations are below the corresponding between-class correlation; the claim that within-class consistency is always higher is not supported by these recomputations.')
    sources(p,['pair_distribution_sensitivity.csv','motif_induced_pair_distributions.csv'])

# Shared content model makes both formats reproducible from the same numbers.
(DEST/'report_pages.json').write_text(json.dumps(PAGES,indent=2,ensure_ascii=False)+'\n')
styles='''@page {size:A4; margin:16mm;} body{font:10pt Arial,sans-serif;color:#111;} article{break-after:page;page-break-after:always;} article:last-child{break-after:auto;} h1{font-size:19pt;margin:0 0 8pt;} h2{font-size:11pt;margin:14pt 0 5pt;} .figures{font-size:10pt;color:#333;} p{line-height:1.35;} table{border-collapse:collapse;width:100%;margin:8pt 0;font-size:9pt;} th,td{padding:4pt 5pt;border-bottom:1px solid #ccc;text-align:left;vertical-align:top;} th{background:#eee;} footer{font-size:8pt;color:#666;margin-top:14pt;}'''
parts=['<!doctype html><html><head><meta charset="utf-8"><title>CISM numerical robustness and uncertainty report</title><style>'+styles+'</style></head><body>']
for i,p in enumerate(PAGES,1):
    parts += ['<article><h1>'+html.escape(p['title'])+'</h1><p class="figures"><b>Relevant figures:</b> '+html.escape(p['figures'])+'</p>']
    for b in p['blocks']:
        if b['type'] in ['paragraph','heading']:
            tag='p' if b['type']=='paragraph' else 'h2';parts.append(f'<{tag}>'+html.escape(b['text'])+f'</{tag}>')
        else:
            parts.append('<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in b['headers'])+'</tr></thead><tbody>')
            for row in b['rows']:parts.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>')
            parts.append('</tbody></table>')
    parts.append(f'<footer>CISM numerical report • 20 November 2026 • {i}/{len(PAGES)} • Google Docs unchanged</footer></article>')
parts.append('</body></html>');(DEST/'CISM_numerical_report.html').write_text('\n'.join(parts))

# Draw only text and table rules: no plots, figure images, or charts.
PAGE_W,PAGE_H=595.276,841.89
MARGIN=42
TEXT_W=PAGE_W-2*MARGIN
measure=TextToPath()
def width(s,size,bold=False):
    return measure.get_text_width_height_descent(str(s),FontProperties(family='DejaVu Sans',size=size,weight='bold' if bold else 'normal'),False)[0]
def wrap(s,maxwidth,size,bold=False):
    maxwidth *= .95  # Leave room for raster/PDF font-metric differences.
    lines=[];line=''
    for word in str(s).split():
        candidate=(line+' '+word).strip()
        if line and width(candidate,size,bold)>maxwidth:
            lines.append(line);line=word
        else:line=candidate
    if line:lines.append(line)
    return lines or ['']

layout=[]
with PdfPages(DEST/'CISM_numerical_report.pdf',metadata={'Title':'CISM numerical robustness and uncertainty report','Author':'CISM revision analysis','Subject':'One analysis per page; numerical estimates only'}) as pdf:
    for idx,p in enumerate(PAGES,1):
        fig=plt.figure(figsize=(PAGE_W/72,PAGE_H/72),dpi=72)
        compact='motif-induced interaction robustness' in p['title']
        y=PAGE_H-MARGIN
        def txt(x,y,text,size=9.1,bold=False,color='#111111'):
            fig.text(x/PAGE_W,y/PAGE_H,str(text),fontsize=size,fontfamily='DejaVu Sans',fontweight='bold' if bold else 'normal',color=color,va='top')
        def lines(text,size=None,bold=False,after=7):
            global y
            if size is None:size=8.8 if compact else 9.1
            for line in wrap(text,TEXT_W,size,bold):
                txt(MARGIN,y,line,size,bold);y-=size*1.35
            y-=after
        lines(p['title'],17,True,9)
        lines('Relevant figures: '+p['figures'],9.4,False,9)
        for b in p['blocks']:
            if b['type']=='paragraph':lines(b['text'])
            elif b['type']=='heading':
                y-=3;lines(b['text'],10.3,True,5)
            else:
                weights=b['widths'] or [1]*len(b['headers'])
                widths=np.array(weights)/sum(weights)*TEXT_W
                size=(8.0 if compact else 8.3) if len(weights)<9 else 7.5
                for ri,row in enumerate([b['headers']]+b['rows']):
                    bold=(ri==0)
                    cells=[wrap(s,w-9,size,bold) for s,w in zip(row,widths)]
                    h=max(map(len,cells))*size*1.3+(4 if compact else 7)
                    x=MARGIN
                    for cell,w in zip(cells,widths):
                        for j,line in enumerate(cell):txt(x+4,y-3-j*size*1.3,line,size,bold)
                        x+=w
                    fig.add_artist(plt.Line2D([MARGIN/PAGE_W,(PAGE_W-MARGIN)/PAGE_W],[(y-h+1)/PAGE_H]*2,transform=fig.transFigure,color='#bcbcbc',linewidth=.4))
                    y-=h
                y-=8
        assert y>45,(idx,p['title'],y)
        txt(MARGIN,28,f'CISM numerical report • 20 November 2026 • {idx}/{len(PAGES)} • Google Docs unchanged',7.3,color='#555555')
        fig.canvas.draw()
        # Check every text object for clipping in page bounds.
        renderer=fig.canvas.get_renderer()
        for obj in fig.texts:
            box=obj.get_window_extent(renderer)
            assert box.x0>=MARGIN-1 and box.x1<=PAGE_W-MARGIN+1,(idx,obj.get_text(),box)
            assert box.y0>=12 and box.y1<=PAGE_H-30,(idx,obj.get_text(),box)
        if idx in [1,5,8,9,12]:fig.savefig(DEST/f'preview_page_{idx:02d}.png',dpi=105)
        pdf.savefig(fig);plt.close(fig)
        layout.append(dict(page=idx,title=p['title'],remaining_bottom_space_points=round(y,2)))
(DEST/'report_audit.json').write_text(json.dumps(dict(status='passed',pages=len(PAGES),one_analysis_per_page=True,plots=0,google_docs_edited=False,layout=layout),indent=2)+'\n')
print(f'Compiled {len(PAGES)} analysis pages to {DEST}')
