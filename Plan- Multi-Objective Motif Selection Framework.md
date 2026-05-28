# Plan: Multi-Objective Motif Selection Framework

## Goal

Implement a multi-objective motif selection framework for identifying discriminative spatial motifs between disease states.

Each motif should receive a final score based on:

1. Discriminative strength / effect size
2. Motif abundance
3. Overall motif prevalence
4. Statistical confidence
5. Dispersion / noisiness
6. Stability across resampling or cross-validation

Disease-state-exclusive motifs should **not** be automatically penalized. A motif may be exclusive to one disease state and still be selected if it is informative and reproducible.

Stability should act as a **multiplicative gate**, because reproducibility is a required condition. The other components should be combined additively using user-defined weights.

---

## Final Objective Function

For each motif `m`, compute:

```text
final_score(m) = stability_gate(m) * additive_score(m)
```

where:

```text
additive_score(m) =
    beta_effect      * effect_score(m)
  + beta_abundance   * abundance_score(m)
  + beta_prevalence  * prevalence_score(m)
  + beta_confidence  * confidence_score(m)
  + beta_dispersion  * dispersion_desirability_score(m)
```

Then normalize by total weight:

```text
additive_score(m) = additive_score(m) / sum(beta_i)
```

The final score is therefore:

```text
final_score(m) =
    stability_gate(m)
    *
    normalized_weighted_sum_of_objectives(m)
```

---

## Conceptual Justification

Use an additive weighted score for components that represent tradeoffs.

For example:
- A motif with slightly lower abundance may still be valuable if it has strong effect size.
- A motif with moderate effect size may still be useful if it is highly prevalent and statistically reliable.

Therefore, effect size, abundance, prevalence, confidence, and dispersion should be combined additively with coefficients.

Use a multiplicative gate for stability because stability is not just another preference. It represents whether the motif is reproducible across resampling or cross-validation. An unstable motif should not rank highly only because it has a strong apparent effect size in one split.

This design allows informative disease-state-exclusive motifs to be selected, while still preventing unstable motifs from dominating the ranking.

---

## Required Inputs

The main function should accept:

a CISM object with an added dataset. (can be found under the outputs of the second tutorial notebook "02_fanmod_and_cism_initizlization")
I want to support both people who run the notebook pipeline without much thought and people who want to use the scoring function more flexibly on their own data.

A patient_class csv mapping patients to disease states.


## Metrics to Compute Per Motif

### 1. Discriminative Effect Score

Use direction-free ROC-AUC effect size.

For each motif:

```python
auc = roc_auc_score(y, motif_values)
effect_score = abs(auc - 0.5) * 2
```

This maps:

```text
AUC = 0.5 -> effect_score = 0
AUC = 1.0 -> effect_score = 1
AUC = 0.0 -> effect_score = 1
```

This is direction-free, meaning a motif enriched in either disease state can receive a high score.

Also store the raw AUC so we can know the direction.

---

### 2. Log Fold-Change

Compute:

```python
logfc = log2((mean_group_1 + pseudocount) / (mean_group_0 + pseudocount))
```

Store:

```python
logfc_group1_vs_group0
abs_logfc
direction
```

Direction should be:

```text
group_1 if logfc > 0
group_0 if logfc < 0
```

This is mostly for interpretation, not necessarily part of the main score.

---

### 3. Motif Abundance Score

Compute mean motif abundance across all patients:

```python
mean_abundance = mean(values)
abundance_score = log1p(mean_abundance)
```

Then normalize abundance scores across motifs to `[0, 1]`.

---

### 4. Overall Prevalence Score

Compute the fraction of patients where the motif appears:

```python
overall_prevalence = mean(values > 0)
```

Do not require the motif to appear in both disease states.

Also compute group-specific prevalence for interpretation:

```python
prevalence_group_0 = mean(values[y == 0] > 0)
prevalence_group_1 = mean(values[y == 1] > 0)
specificity = abs(prevalence_group_1 - prevalence_group_0)
```

Important:

`overall_prevalence` may be included in the score, but `specificity` should only be reported, not penalized.

---

### 5. Statistical Confidence Score

For each motif, compute a group-comparison p-value.

Use Mann-Whitney U test by default:

```python
mannwhitneyu(group_0_values, group_1_values, alternative="two-sided")
```

Then correct across all motifs using Benjamini-Hochberg FDR:

```python
qvalues = multipletests(pvalues, method="fdr_bh")
```

Define:

```python
fdr_confidence = 1 - qvalue
```

Then normalize to `[0, 1]`.

---

### 6. Dispersion / Noise Score

Compute dispersion using the coefficient of variation on log-transformed values:

```python
transformed = log1p(values)
dispersion = std(transformed) / (mean(transformed) + epsilon)
```

Since lower dispersion is better, convert it to desirability:

```python
dispersion_desirability = 1 - normalize_01(dispersion)
```

This should gently reward motifs that are not extremely noisy.

---

### 7. Stability Gate

Stability should be used as a multiplicative gate.

Use:

```python
gate = min(1.0, stability / tau) ** gamma
```

Default:

```python
tau = 0.6
gamma = 2.0
```

Examples:

```text
stability = 0.60 -> gate = 1.00
stability = 0.45 -> gate = 0.56
stability = 0.30 -> gate = 0.25
stability = 0.15 -> gate = 0.06
```

This is a soft gate, not a hard cutoff.

---


## Main Design Decisions

1. Disease-state-exclusive motifs are allowed.
2. Stability is multiplicative because it is a reproducibility requirement.
3. Other metrics are additive because they represent tradeoffs.
4. Weights allow the user to control the relative importance of each additive component.
5. weights should be normalized to sum to 1 for interpretability and be chosen using optuna or any other optimization tool.
6. any evaluation on the dataset should be done in a random forest framework like the one already present under CISM in a leave one out cross validation manner.
7. Redundancy is handled after scoring to preserve interpretability.
8. All normalization is performed across motifs.
9. Predictive validation must avoid feature-selection leakage.

## caveats

1. consult me for anything unclear or if you want to deviate from the plan for any reason. I am open to changes but let's discuss first.
2. avoid code duplication. if assets already exist in other places in the codebase, reuse or adapt them. asses if any such adaptation affects the API and notify me if it affects any othe file or notebook that uses the asset.
3. make sure to write tests for any new functions you create, and ensure that existing tests still pass. If you modify existing functions, check if there are tests for them and update the tests accordingly.
4. document any new functions with docstrings, and update any relevant documentation to reflect the new functionality.
