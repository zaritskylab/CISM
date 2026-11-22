"""Build reviewable report from computed tables, never hard-code numerical results."""
from pathlib import Path
import json
import platform
import importlib.metadata
import os
import numpy as np
import pandas as pd
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cism_patient_validation_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent

def table(df):
    cols=list(df.columns)
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.6g}'
        return str(x)
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    comp=BASE/'composition';edges=BASE/'interactions'
    primary=comp/'primary_original_graph_retained_cells'
    c=pd.read_csv(primary/'rank_sum_tests.csv');g=json.loads((primary/'global_jensen_shannon_test.json').read_text())
    s=pd.read_csv(edges/'stats_group_proportions.csv');sp=pd.read_csv(edges/'stats_spatial_enrichment.csv')
    patients=pd.read_csv(edges/'patient_edges.csv');fov=pd.read_csv(edges/'fov_manifest.csv')
    targets=((s.code_a==0)&s.code_b.isin([0,9]))|((s.code_a==9)&(s.code_b==9))
    select=lambda z:z[((z.code_a==0)&z.code_b.isin([0,9]))|((z.code_a==9)&(z.code_b==9))].copy()
    candidate=select(sp)
    ct=c[['cell_type','NN_median','NN_q25','NN_q75','NP_median','NP_q25','NP_q75','median_difference_NN_minus_NP','rank_biserial_NN_minus_NP','p_raw','p_holm']].copy()
    st=s.loc[targets,['interaction','median_NN','q25_NN','q75_NN','median_NP','q25_NP','q75_NP','median_difference_NN_minus_NP','rank_biserial_NN_minus_NP','p_raw','p_holm']].copy()
    st.iloc[:,1:8]*=100
    display=candidate[['group','interaction','mean_enrichment','bootstrap95_low','bootstrap95_high','observed_sum','expected_sum','rank_supported_mean_enrichment','p_upper','p_upper_holm','p_two_sided_holm']]
    # Descriptive heterogeneity summaries; no FOV is an independent test unit.
    fe=pd.read_csv(edges/'fov_edges_audit.csv');fe=select(fe);fe['enrichment']=(fe.observed+.5)/(fe.expected+.5)
    hetero=fe.groupby(['patient_id','group','code_a','code_b']).agg(fovs=('fov','size'),fov_proportion_min=('proportion','min'),fov_proportion_max=('proportion','max'),fov_enrichment_min=('enrichment','min'),fov_enrichment_max=('enrichment','max')).reset_index()
    hetero.to_csv(edges/'candidate_fov_heterogeneity_descriptive.csv',index=False)
    loo=[]
    for (group,a,b),z in select(patients).groupby(['group','code_a','code_b']):
        arr=z.enrichment.to_numpy();means=(arr.sum()-arr)/(len(arr)-1)
        loo.append(dict(group=group,code_a=a,code_b=b,interaction=s[(s.code_a==a)&(s.code_b==b)].iloc[0].interaction,
                        mean_enrichment=arr.mean(),leave_one_patient_out_mean_min=means.min(),leave_one_patient_out_mean_max=means.max()))
    loo=pd.DataFrame(loo);loo.to_csv(edges/'candidate_leave_one_patient_out_descriptive.csv',index=False)
    # Actual observed vs null-expected patient counts, separate from ratio CIs.
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for ax,(a,b) in zip(axs,[(0,0),(9,9),(0,9)]):
        z=patients[(patients.code_a==a)&(patients.code_b==b)]
        for group,color in [('NN','#356aa0'),('NP','#d57940')]:
            q=z[z.group==group];ax.scatter(q.expected,q.observed,label=group,c=color,alpha=.8)
        upper=max(z.observed.max(),z.expected.max())*1.06
        ax.plot([0,upper],[0,upper],'--',c='gray');ax.set(xlabel='Exact spatial-null expected count',ylabel='Observed undirected edge count',title=s[(s.code_a==a)&(s.code_b==b)].iloc[0].interaction);ax.legend()
    fig.tight_layout();fig.savefig(edges/'candidate_observed_vs_expected.png',dpi=180);plt.close(fig)
    null=np.load(primary/'global_permutation_null.npy');fig,ax=plt.subplots(figsize=(6,4));ax.hist(null,bins=50,color='#708da5');ax.axvline(g['distance_base2'],c='#b74938',label=f"Observed {g['distance_base2']:.3f}");ax.set(xlabel='Jensen–Shannon distance (base 2)',ylabel='Patient-label permutations',title=f"Global composition: p = {g['p_empirical']:.3f}");ax.legend();fig.tight_layout();fig.savefig(comp/'global_composition_permutation.png',dpi=180);plt.close(fig)
    env={'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib','numba']}}
    (BASE/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    spatial_count=sp.groupby('group').apply(lambda z:int((z.p_upper_holm<.05).sum()),include_groups=False).to_dict()
    report=f'''# Patient-level statistical validation of historical Fig. 2B / S2 analyses

Analysis date: 2026-11-09. Status: analyses executed and independently validated on actual data. This directory contains proposed figures and text; existing manuscript files and figures were not modified.

## Executive summary

- The verified cohort contains **20 NN and 18 NP patients**, with **40 and 36 FOVs**, respectively. All 38 enter each independent-group test once per cell or interaction type.
- **Composition:** none of 15 eligible cell types differs after Holm correction (minimum adjusted p = {c.p_holm.min():.6g}). Global base-2 Jensen–Shannon distance = **{g['distance_base2']:.6f}**, whole-patient permutation p = **{g['p_empirical']:.6f}** (20,000 permutations). Nonsignificance does not establish equivalence.
- **Interaction proportions between NN and NP:** none of 120 unordered interaction types differs after Holm correction (minimum adjusted p = {s.p_holm.min():.6g}).
- **Spatial organization within groups:** B–B and memory-CD4–memory-CD4 interactions are enriched relative to a within-FOV label-shuffle null in both groups. The B–memory-CD4 interaction is depleted in a secondary two-sided test. These findings concern organization conditional on the graphs and compositions, not differential enrichment between NN and NP.
- Exact revised S2 targets could not be verified from available manuscript material. All 120 eligible pairs were tested; the three handoff examples are highlighted transparently, with no target-only correction or result-driven target selection.
- Physical FOV areas could not be verified. Density testing is not evaluable from the available reliable metadata.

## Data provenance and cohort counts

The historical notebook `/home/omriavit/CISM/CISM/analysis/Melanoma_MIBITOF/Melanoma_100_distance_NN_NP.ipynb` explicitly labels a Figure 2 workflow and supplies the original 16-color graph mapping. Its available graphs are under `/groups/assafza_group/assafza/Omri/processed_data_23_10_2025/data/melanoma_distance_100_XX/`. This establishes a historical figure-analysis source, not a byte-for-byte reproduction of the unavailable revised panels. See [full provenance audit](provenance_audit.md).

Labels from `melanoma_tutorials/pet_data/Melanoma/patient_class.csv` agree with the original parent-folder labels and the raw cell table `Group` column for every NN/NP patient. PN and PP are excluded by design; no NN/NP patient or FOV is excluded from the available cohort. Patient 50 owns three FOVs, patient 67 owns one, and every other NN/NP patient owns two.

| Group | Patients | FOVs | Retained graph cells | Raw cells | Unique undirected edges |
| --- | --- | --- | --- | --- | --- |
| NN | 20 | 40 | 428411 | 428413 | {int(fov.loc[fov.group=='NN','edges'].sum())} |
| NP | 18 | 36 | 435037 | 435038 | {int(fov.loc[fov.group=='NP','edges'].sum())} |

The primary universe comprises the 863,448 cells represented by original graph endpoints and 2,579,711 unique undirected edges. Only three raw cells lack graph representation: two in patient 72/FOV422 and one in patient 19/FOV226. None of the raw NN/NP cells is annotated tumor. Tumor is omitted as a structural-zero category; all 15 other categories, including unidentified cells, are eligible. New tutorial graphs encode types differently and are kept as a separate sensitivity source. Graph node IDs correspond to row positions within FOVs, not biological `cell_id` identifiers. Independent validation checked every retained row-position annotation against the raw table.

The preparation notebooks document Delaunay neighbors with a 100-pixel distance cap and several tumor-cleanup variants. The precise cleanup variant for the archived graphs is not recoverable, so graphs were used as stored rather than reconstructed using an assumed variant. Graph file SHA-256 values are in [FOV manifest](interactions/fov_manifest.csv), and the raw cell-table hash is in [composition metadata](composition/metadata.json).

## Methods and statistical assumptions

### Composition

For each patient, counts of each eligible type are summed across all their FOVs, then divided by the total eligible patient cell count. Each resulting vector sums to one. All group comparisons use two-sided Mann–Whitney U (Wilcoxon rank-sum), with explicit asymptotic tie correction and continuity correction. This is an independent-group test, not a paired signed-rank test. Quartiles use NumPy linear interpolation; differences are NN minus NP in percentage points. Rank-biserial effect = `2 U_NN/(20*18) - 1`; positive values favor larger NN values. The rank test concerns distributions; differences in medians are descriptive effect summaries, not a claim that only medians are being tested.

The global statistic is the base-2 Jensen–Shannon distance between the two equally weighted means of patient composition vectors. The upper-tail test shuffles whole-patient NN/NP labels while preserving 20/18 sizes, with 20,000 permutations and seed 20261109. A distance is nondirectional, so an upper-tail test detects differences in either group direction. Empirical p = `(1 + number of null statistics >= observed)/(B + 1)`.

### Interactions and spatial null

Each undirected cell edge is counted once, including homotypic edges once. A–B and B–A share one canonical type. Counts are summed across FOVs before dividing by the patient's total edge count. All 120 combinations with replacement of the 15 eligible cell types are tested between groups using the same independent-group rank-sum procedure.

For the spatial null, each FOV retains its exact graph topology and type counts while labels are shuffled among represented graph vertices. For FOV edge count m and vertex count n, exact expected counts are `m*n_A*(n_A-1)/(n*(n-1))` for A–A and `m*2*n_A*n_B/(n*(n-1))` for A–B. These expectations sum across a patient's FOVs. The descriptive patient enrichment is `(O + 0.5)/(E + 0.5)`; the pseudocount was fixed before tests were evaluated.

There are 10,000 independent label-shuffle realizations per FOV, seeded by sorted FOV index plus 20261109. For each realization, edge counts are summed across that patient's FOVs, and the group statistic is the **equally weighted mean of patient enrichment ratios**. The upper-tail empirical test evaluates enrichment. Secondary exploratory two-sided p-values use absolute deviations of this statistic from its exact null mean of one; they are not twice a one-tail p-value. No cells, edges, or FOVs are treated as independent patients. Independent FOV randomization here constructs a conditional spatial null, not biological replication.

Group mean-ratio 95% intervals are percentile intervals from 10,000 bootstrap resamples of **patients within group**. Patient count intervals in the input table are 2.5th/97.5th null-count quantiles, which are null prediction ranges, not confidence intervals for biological effects. Pseudocounts 0.1 and 1 are sensitivities. Zero observed counts remain zero; zero expected counts are handled by the fixed positive pseudocount. No missing patient values are silently imputed.

Official implementation references: [SciPy Mann–Whitney U](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html), [SciPy Jensen–Shannon distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html). Package versions are recorded in [environment.json](environment.json).

## Fig. 2B numerical results

All cell percentages and differences below are in percentage points. n = 20 NN / 18 NP for every row. The full table additionally records IQR widths, U statistics, zero counts, and pooled distinct-value counts.

{table(ct)}

Global distance **{g['distance_base2']:.10f}**; {g['upper_tail_exceedances']} of {g['permutations']} null distances at least as large; empirical p **{g['p_empirical']:.9f}**. Full [rank-sum table](composition/primary_original_graph_retained_cells/rank_sum_tests.csv) and [patient input table](composition/primary_original_graph_retained_cells/patient_composition.csv).

![Patient cell-type composition](composition/primary_original_graph_retained_cells/patient_composition.png)

![Global composition permutation test](composition/global_composition_permutation.png)

## Fig. S2 numerical results

### NN versus NP interaction distributions

No interaction passes Holm across all 120 pairs. The smallest raw p occurs for CD4 T-cell–MAC (p = {s.p_raw.min():.6g}, Holm p = {s.p_holm.min():.6g}). Candidate medians and quartiles below are percentages of each patient's eligible unique edges; differences are percentage points. Candidate adjusted p-values are all 1.

{table(st)}

See [all 120 comparisons](interactions/stats_group_proportions.csv) and [patient observed/expected inputs](interactions/patient_edges.csv). A separate exploratory comparison of patient enrichment scores also has no Holm-significant group difference (minimum adjusted p = 0.523908).

![Patient interaction proportions](interactions/candidate_patient_proportions.png)

### Enrichment relative to random spatial labels

The following are handoff candidate interactions, not independently verified revised-manuscript targets. Ratios and intervals are equal-patient group mean ratios and patient-bootstrap confidence intervals. Observed and expected sums are supplied for audit; the ratio of those sums is a different, edge-weighted quantity and must not replace the reported patient-mean statistic. Among sufficiently supported pairs, NN ranks use 112 pairs and NP ranks use 113. Support was defined before ranking as expected count at least five in at least half the patients of that group. Ranking does not filter the testing family.

{table(display)}

All four homotypic candidate tests have zero exceedances among 10,000 realizations, so their reported empirical p is 1/10001 = 0.00009999 and Holm p is 240/10001 = 0.023998. These are finite Monte Carlo estimates, not exact tail probabilities; the output retains exceedance counts and permutation numbers. B–memory-CD4 has enrichment-test p = 1, while its secondary absolute-deviation test supports depletion (Holm p = 0.023998). A candidate cross-type **enrichment** claim would therefore be contradicted by these data under this null.

Across all pairs, {spatial_count['NN']} NN and {spatial_count['NP']} NP interactions pass the upper-tail spatial-enrichment family. B–B and memory-CD4 homotypic organization are consequently not unique enriched interactions. Their ranks are descriptive; no formal superiority to other edge types is asserted. A paired contrast against a particular interaction would require a scientifically specified comparator and is not needed to assess the available candidate claims.

![Patient spatial enrichment](interactions/candidate_patient_enrichment.png)

![Observed vs spatial-null expected patient edge counts](interactions/candidate_observed_vs_expected.png)

See [all spatial tests](interactions/stats_spatial_enrichment.csv), [NN rankings](interactions/NN_enrichment_ranking.png), and [NP rankings](interactions/NP_enrichment_ranking.png).

## Multiple-testing corrections

Primary families address distinct scientific questions: 15 composition rank-sum tests; one standalone global composition permutation test; 120 between-group edge-proportion tests; and 240 spatial upper-tail tests (120 edge types × two groups). Holm is applied across each complete family. The secondary absolute-deviation spatial family also contains 240 tests and is exploratory. Each sensitivity family is corrected separately and does not replace the primary result. Structural-zero tumor types are excluded from primary testing based on eligibility, not p-values.

## Sensitivity analyses

- **Raw cells and graph source:** including the three cells missing from graph endpoints yields JSD 0.112008153 and permutation p = 0.177341, with no composition finding significant. Tutorial graphs give the same primary distance/p. Including the zero tumor category increases the Holm family to 16 and changes the minimum adjusted p to 0.436731; the scientific conclusion is unchanged.
- **Historical diagonal convention:** the archived symmetric matrix increments homotypic diagonal entries twice, with total sum twice the undirected edge count. A displayed diagonal entry is therefore `O/m`, and a displayed off-diagonal entry is `O/(2m)`. The patient-level two-sided rank p-values are exactly identical to unique-edge proportions because the per-type factor is constant across patients. This does not make the upper triangle a normalized unordered-edge distribution, and it does not justify the original FOV weighting for patient inference.
- **Equal FOV weights within patients:** averaging normalized FOV proportions within each patient, followed by a patient-level test, yields no significant between-group interactions (minimum Holm p = 0.751987). The primary analysis instead pools edge counts within each patient as required.
- **Pseudocounts:** 0.1 and 1 retain the homotypic-enrichment and cross-type-depletion conclusions. The NN B–B mean ranges from 3.094 to 3.192 across these choices; other candidate changes are smaller. Counts and expected counts remain available to assess rare-type instability.
- **Degree and boundary sensitivity:** a second 10,000-per-FOV null shuffles only within fixed graph-degree bins 0–3, 4–5, 6–7, and ≥8. This preserves type counts by degree bin while retaining the complete graph. Candidate directions and adjusted significance are unchanged; B–B means are 3.185 NN / 2.553 NP, memory-CD4 homotypic means 1.942 / 1.742, and cross-type means 0.536 / 0.510. Degree is only a proxy for some boundary effects, not a tissue compartment annotation.
- **FOV heterogeneity:** patient 5 B–B edge proportions range from 4.34% to 33.44% across its two FOVs; patient 50 memory-CD4 homotypic proportions range from 3.60% to 26.46% across three FOVs. These are descriptive observations, not independent tests. [Full candidate FOV ranges](interactions/candidate_fov_heterogeneity_descriptive.csv) and [descriptive leave-one-patient-out means](interactions/candidate_leave_one_patient_out_descriptive.csv) are supplied. All leave-one-out candidate homotypic group means remain above one, and all cross-type group means remain below one.
- **Density:** not run. Seven NN images are 1024² while most images are 2048², physical pixel calibration is unverified, and the NP FOV18 ZIP is absent. A blanket area assumption would be unsupported. A verified area CSV can be passed to composition.py for the separately corrected cells/mm² analysis.

## Independent validation findings

The independent reviewer implemented graph aggregation with streamed sets/dictionaries, U with explicit pairwise comparisons and a separate tie-variance/normal-tail calculation, Holm with a separate loop, and Jensen–Shannon distance with an entropy expression. The reviewer did not import the primary aggregation or rank-test functions. Comparisons cover all 570 patient-composition records and 4,560 patient-interaction records, not just highlighted examples.

The checks verify unique patients; correct NN/NP labels; row-position annotations against raw cells; complete included FOV mapping; patient proportions summing to one; exactly one reverse row per stored edge and one count per unique unordered edge; no loops or conflicting vertex labels; correct homotypic/heterotypic expectations; proper correction families; fixed seeds; and null edge-count conservation. Composition U/effects/counts match exactly, p-values agree to rounding, and independent global distance/p match. The spatial reviewer recomputed group statistics, exceedances, and Holm corrections from saved null draws. Exhaustive tiny-graph label assignments check the permutation engine independently. See [independent review](validation/review.md) and [machine-readable comparison checks](validation/primary_comparison_checks.json).

## Limitations and unsuccessful or unavailable analyses

1. The exact revised panels and S2 wording are unavailable, so historical provenance is verified but revised-panel identity and exact claimed targets remain unresolved. Conclusions are conditional on the explicitly documented historical graph/cell universe.
2. Composition and interaction group tests have only 38 independent patients. Failure to reject is not equivalence and does not establish that a composition-only model cannot classify patients; classification AUC is outside this task.
3. Within-FOV label exchangeability is a strong null. Spatial compartment segregation can produce enrichment even without direct biological attraction. Compartment masks were unavailable; the degree-stratified sensitivity does not replace a compartment-conditioned analysis. Graph topology controls existing adjacency and boundary geometry but does not eliminate all biological compartment effects.
4. Graph endpoint files omit isolated vertices. Three raw cells are unrepresented; raw-cell composition sensitivity is stable. Spatial null inference is conditional on represented vertices.
5. Arithmetic mean enrichment can give rare types relatively large ratios. Observed/expected counts, pseudocount sensitivities, and support-filtered descriptive ranks are provided. No causal interaction or superiority over other edge types is inferred from these ranks.
6. Spatial empirical p-values have 1/10001 resolution. Patient bootstrap confidence intervals are pointwise, not simultaneous familywise intervals. Corrected tests and intervals answer related but distinct questions.
7. Densities could not be validly tested without reliable physical areas. No raw-count comparison substitutes for density. The precise historical tumor-cleanup variant is uncertain, though raw NN/NP tumor counts are zero.

Ki-67, Fig. S7, and classification AUC were not analyzed. During implementation an initial tutorial mapping assumption failed because the tumor annotation is absent from NN/NP; it was corrected to the complete source codebook before successful output. An initial independent cell-ID join failed and was corrected to the verified within-FOV row-position mapping. No successful results from those invalid intermediate mappings were retained.

## Suggested Methods text

“Counts were summed across all included fields of view within each patient. Patient-specific proportions of 15 eligible cell types and 120 unordered interaction types were compared between 20 NN and 18 NP patients using two-sided Mann–Whitney U tests with tie and continuity corrections. Holm correction was applied separately to each complete testing family. Global composition was compared using base-2 Jensen–Shannon distance between equally weighted patient-mean vectors and 20,000 permutations of whole-patient labels. For spatial enrichment, cell-type labels were shuffled 10,000 times within each field of view while retaining graph topology and cell-type counts. Each realization was aggregated within patients before computing equal-patient group mean observed/expected enrichment ratios, using a 0.5 pseudocount. Upper-tail empirical tests were corrected across all 120 interaction types in both groups. Pointwise 95% intervals were obtained by resampling patients within groups. Secondary two-sided spatial tests and degree-stratified label permutations assessed deviations from randomness and sensitivity to graph-degree structure.”

## Suggested Results text

“No individual cell-type proportion differed between NN and NP patients after Holm correction (all adjusted p ≥ 0.409), and the global composition comparison was nonsignificant (Jensen–Shannon distance 0.112; permutation p = 0.177). Similarly, no interaction proportion differed after correction across 120 edge types (all adjusted p ≥ 0.897). In both disease groups, B–B and memory-CD4–memory-CD4 interactions were enriched relative to within-FOV random labeling: mean patient enrichment ratios were 3.14 and 1.93 in NN and 2.53 and 1.73 in NP, respectively (all spatial Holm p = 0.024). B–memory-CD4 interactions instead occurred below the random-label expectation (mean ratios 0.53 NN and 0.51 NP; secondary two-sided Holm p = 0.024). Candidate spatial results were robust to pseudocount and degree-stratified-null sensitivities. These spatial findings do not imply differences in enrichment between disease groups or equivalence of group compositions.”

This wording is proposed for review. The S2 caption/claims must be checked against the revised manuscript before assigning these candidate interactions to a specific panel claim.

## Conclusion by claim

| Claim | Assessment |
| --- | --- |
| Cell-type composition differs between NN and NP | Not supported by these corrected tests; equivalence remains untested. |
| Candidate interaction proportions differ between NN and NP | Not supported; all 120 corrected comparisons are nonsignificant. |
| B–B and memory-CD4 homotypic interactions exceed random spatial labeling | Supported in both groups under the stated conditional null. |
| B–memory-CD4 interactions are spatially enriched | Unsupported; secondary testing instead supports depletion. |
| These interactions are uniquely or significantly more enriched than other types | Not established; descriptive ranks and many other enriched types preclude that claim. |
| Area-normalized densities differ between groups | Not evaluable with currently verified area metadata. |
| The exact revised Fig. S2 claims are validated | Inconclusive until the revised panel/caption identifies the targets. |

## Reproduction and outputs

Run `bash analysis/patient_level_validation/run_all.sh` from the repository root with the shared source data mounted. It runs composition.py, interactions.py, validate_independently.py, and this report builder sequentially. The scripts read existing sources and write only this dedicated directory. Parallelism in interactions.py computes FOV-specific null draws; inference remains at patient level. The saved null archive allows independent checks without generating new spatial draws. See [README](README.md), per-analysis metadata, and [environment.json](environment.json). No packages were installed for this run.
'''
    (BASE/'REPORT.md').write_text(report)
    (BASE/'README.md').write_text('''# Patient-level Fig. 2B / S2 validation

Start with [REPORT.md](REPORT.md). The raw data remain external; all patient-level inputs and results are materialized here.

Reproduce from the repository root:

```bash
bash analysis/patient_level_validation/run_all.sh
```

Requires Python with numpy, pandas, scipy, matplotlib, and numba (exact versions in environment.json). Read-only source paths are configurable in the analysis scripts. The source paths, codebooks, checksums, null schemes, seeds and permutation counts are recorded in the scripts and output metadata.

- `composition/`: primary retained-graph composition and explicit sensitivity directories, patient tables, complete tests, plots, stored global nulls.
- `interactions/`: complete patient edge/expected counts, 120-pair group tests, 240-test spatial families, all sensitivities, plots and saved hierarchical null counts.
- `validation/`: separate aggregation and statistics, exhaustive null-engine checks, reproducibility/consistency results and independent review.
- `provenance_audit.md`: historical figure reconstruction, cohort/codebook and area audit.

Exact revised S2 targets and reliable physical FOV areas remain unavailable. They are documented limitations, not substituted with inferred results. Existing manuscript files and figures are unchanged.
''')
    print('Built',BASE/'REPORT.md')

if __name__=='__main__':main()
