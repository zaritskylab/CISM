from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from cism.cism import AnalyzeMotifsResult, DiscriminativeMotifs


@dataclass(frozen=True)
class MotifSelectionWeights:
    """Weights for the additive part of multi-objective motif scoring."""

    effect: float = 1.0
    abundance: float = 1.0
    prevalence: float = 1.0
    confidence: float = 1.0
    dispersion: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "effect": float(self.effect),
            "abundance": float(self.abundance),
            "prevalence": float(self.prevalence),
            "confidence": float(self.confidence),
            "dispersion": float(self.dispersion),
        }

    def total(self) -> float:
        total = sum(self.as_dict().values())
        if total <= 0:
            raise ValueError("At least one motif selection weight must be positive.")
        return total


@dataclass(frozen=True)
class StabilityGateConfig:
    """Soft multiplicative gate applied to the additive motif score."""

    tau: float = 0.6
    gamma: float = 2.0

    def apply(self, stability: float) -> float:
        if self.tau <= 0:
            raise ValueError("tau must be positive.")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        return float(min(1.0, max(0.0, stability) / self.tau) ** self.gamma)


@dataclass(frozen=True)
class SoftMotifSelectionConfig:
    """Configuration for multi-objective soft motif selection."""

    labels: list[str]
    top_k: int
    weights: MotifSelectionWeights = field(default_factory=MotifSelectionWeights)
    gate: StabilityGateConfig = field(default_factory=StabilityGateConfig)
    fanmod_p_value_threshold: float | None = 0.05
    pseudocount: float = 1e-9
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if len(self.labels) != 2:
            raise ValueError("Soft motif selection currently supports exactly two labels.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.pseudocount <= 0:
            raise ValueError("pseudocount must be positive.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")


@dataclass
class SoftMotifSelectionResult:
    """Scored motifs and selected motif ids for downstream CISM analysis."""

    scores: pd.DataFrame
    selected_motif_ids: list[Any]
    patient_motif_matrix: pd.DataFrame
    labels: list[str]

    def to_inference_feature_config(self):
        """Return an InferenceFC that reuses the selected motif ids."""
        from cism.cism import InferenceFC

        return InferenceFC(labels=self.labels, motifs_ids=self.selected_motif_ids)


@dataclass
class SoftMotifLOOCVResult:
    """Leakage-safe LOOCV validation output for selected soft motifs."""

    analyze_result: AnalyzeMotifsResult
    fold_score_tables: dict[str, pd.DataFrame]
    selected_features_by_patient: dict[str, list[Any]]

    @property
    def results(self) -> pd.DataFrame:
        return self.analyze_result.results

    def get_metrics(self):
        return self.analyze_result.get_metrics()

    def get_roc_auc_score(self) -> float:
        return self.analyze_result.get_roc_auc_score()


def build_patient_motif_matrix(
    motifs_df: pd.DataFrame,
    patient_ids: list[Any] | pd.Index | None = None,
    motif_col: str = "ID",
    patient_col: str = "Patient_uId",
) -> pd.DataFrame:
    """Build a patient-by-motif frequency matrix using CISM's motif frequency formula."""
    required = {patient_col, motif_col, "FOV", "Count", "Freq"}
    missing = required.difference(motifs_df.columns)
    if missing:
        raise ValueError(f"motifs_df is missing required columns: {sorted(missing)}")

    if patient_ids is None:
        patient_ids = pd.Index(motifs_df[patient_col].dropna().unique(), name=patient_col)
    else:
        patient_ids = pd.Index(patient_ids, name=patient_col)

    if motifs_df.empty:
        return pd.DataFrame(index=patient_ids)

    data = motifs_df[[patient_col, motif_col, "FOV", "Count", "Freq"]].copy()
    data = data[data["Freq"] > 0]
    data["_total_subgraphs"] = data["Count"] / data["Freq"]

    total_counts = data.groupby([patient_col, motif_col], observed=True)["Count"].sum()
    total_subgraphs = (
        data.drop_duplicates([patient_col, motif_col, "FOV"])
        .groupby([patient_col, motif_col], observed=True)["_total_subgraphs"]
        .sum()
    )
    grouped = pd.concat(
        [total_counts.rename("total_count"), total_subgraphs.rename("total_subgraphs")],
        axis=1,
    ).reset_index()
    grouped["value"] = grouped["total_count"] / grouped["total_subgraphs"]

    matrix = grouped.pivot(index=patient_col, columns=motif_col, values="value")
    matrix = matrix.reindex(index=patient_ids).fillna(0.0)
    matrix.columns.name = None
    return matrix


def score_soft_motifs(
    motifs_df: pd.DataFrame,
    patient_class_df: pd.DataFrame,
    config: SoftMotifSelectionConfig,
    stability: pd.Series | dict[Any, float] | None = None,
    motif_col: str = "ID",
) -> SoftMotifSelectionResult:
    """Score motifs with the multi-objective soft motif selection objective."""
    local_classes = _prepare_patient_classes(patient_class_df, config.labels)
    local_motifs = _filter_motifs(motifs_df, config.fanmod_p_value_threshold)
    matrix = build_patient_motif_matrix(local_motifs, patient_ids=local_classes.index, motif_col=motif_col)
    scores = _score_matrix(matrix, local_classes[DiscriminativeMotifs.PATIENT_CLASS], config)

    if stability is None:
        scores["stability"] = 1.0
    else:
        stability_series = pd.Series(stability, dtype=float)
        scores["stability"] = scores.index.to_series().map(stability_series).fillna(0.0).astype(float)

    scores["stability_gate"] = scores["stability"].map(config.gate.apply)
    scores["final_score"] = scores["stability_gate"] * scores["additive_score"]
    scores = scores.sort_values(["final_score", "additive_score"], ascending=False)
    selected = scores.head(config.top_k).index.tolist()

    return SoftMotifSelectionResult(
        scores=scores.reset_index().rename(columns={"index": motif_col, "motif_id": motif_col}),
        selected_motif_ids=selected,
        patient_motif_matrix=matrix,
        labels=list(config.labels),
    )


def score_soft_motifs_from_discriminator(discriminator, config: SoftMotifSelectionConfig) -> SoftMotifSelectionResult:
    """Score motifs from a TissueStateDiscriminativeMotifs instance."""
    patient_class_df = discriminator.get_patients_class(config.labels)
    stability = compute_loocv_selection_stability(discriminator.cism.motifs_dataset, patient_class_df, config)
    return score_soft_motifs(
        motifs_df=discriminator.cism.motifs_dataset,
        patient_class_df=patient_class_df,
        config=config,
        stability=stability,
    )


def compute_loocv_selection_stability(
    motifs_df: pd.DataFrame,
    patient_class_df: pd.DataFrame,
    config: SoftMotifSelectionConfig,
    motif_col: str = "ID",
) -> pd.Series:
    """Estimate motif stability as training-fold top-k selection frequency under LOOCV."""
    local_classes = _prepare_patient_classes(patient_class_df, config.labels)
    selected_counts: defaultdict[Any, int] = defaultdict(int)
    eligible_counts: defaultdict[Any, int] = defaultdict(int)

    for held_out_patient in local_classes.index:
        train_classes = local_classes.drop(index=held_out_patient)
        train_motifs = motifs_df[motifs_df["Patient_uId"].isin(train_classes.index)]
        train_motifs = _filter_motifs(train_motifs, config.fanmod_p_value_threshold)
        matrix = build_patient_motif_matrix(train_motifs, patient_ids=train_classes.index, motif_col=motif_col)
        if matrix.empty:
            continue

        fold_scores = _score_matrix(matrix, train_classes[DiscriminativeMotifs.PATIENT_CLASS], config)
        top_motifs = set(fold_scores.sort_values("additive_score", ascending=False).head(config.top_k).index)

        for motif_id in fold_scores.index:
            eligible_counts[motif_id] += 1
        for motif_id in top_motifs:
            selected_counts[motif_id] += 1

    stability = {
        motif_id: selected_counts[motif_id] / eligible_count
        for motif_id, eligible_count in eligible_counts.items()
        if eligible_count > 0
    }
    return pd.Series(stability, dtype=float)


def evaluate_soft_motif_selection_loocv(
    discriminator,
    config: SoftMotifSelectionConfig,
    random_state: int = 0,
) -> SoftMotifLOOCVResult:
    """Evaluate soft motif selection in a leakage-safe leave-one-patient-out loop."""
    patient_class_df = discriminator.get_patients_class(config.labels)
    local_classes = _prepare_patient_classes(patient_class_df, config.labels)
    full_motifs_df = discriminator.cism.motifs_dataset
    analyze_rows = []
    fold_score_tables = {}
    selected_by_patient = {}

    for held_out_patient in local_classes.index:
        train_classes = local_classes.drop(index=held_out_patient)
        test_classes = local_classes.loc[[held_out_patient]]
        train_motifs = full_motifs_df[full_motifs_df["Patient_uId"].isin(train_classes.index)]
        test_motifs = full_motifs_df[full_motifs_df["Patient_uId"].isin(test_classes.index)]

        train_result = score_soft_motifs(
            motifs_df=train_motifs,
            patient_class_df=train_classes,
            config=config,
            stability=None,
        )
        selected = train_result.selected_motif_ids
        fold_score_tables[str(held_out_patient)] = train_result.scores
        selected_by_patient[str(held_out_patient)] = selected

        if not selected:
            raise ValueError(f"No motifs selected for held-out patient {held_out_patient}.")

        x_train = train_result.patient_motif_matrix.reindex(columns=selected, fill_value=0.0)
        y_train = train_classes[DiscriminativeMotifs.PATIENT_CLASS]
        x_test = build_patient_motif_matrix(test_motifs, patient_ids=test_classes.index).reindex(
            columns=selected, fill_value=0.0
        )
        y_test = test_classes[DiscriminativeMotifs.PATIENT_CLASS].iloc[0]

        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)[0]
        prob = clf.predict_proba(x_test)

        tp = int(pred == y_test and y_test == config.labels[0])
        tn = int(pred == y_test and y_test == config.labels[1])
        fn = int(pred == config.labels[1] and y_test == config.labels[0])
        fp = int(pred == config.labels[0] and y_test == config.labels[1])

        analyze_rows.append(
            (
                tp,
                tn,
                fn,
                fp,
                len(selected),
                prob,
                y_test,
                pred,
                clf.classes_,
                zip(selected),
                (None, x_test),
            )
        )

    analyze_result = AnalyzeMotifsResult(
        analyze_results=analyze_rows,
        patients_ids=local_classes.index.tolist(),
        labels=list(config.labels),
    )
    return SoftMotifLOOCVResult(
        analyze_result=analyze_result,
        fold_score_tables=fold_score_tables,
        selected_features_by_patient=selected_by_patient,
    )


def _prepare_patient_classes(patient_class_df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    if DiscriminativeMotifs.PATIENT_CLASS not in patient_class_df.columns:
        raise ValueError(f"patient_class_df must include a '{DiscriminativeMotifs.PATIENT_CLASS}' column.")
    local = patient_class_df[patient_class_df[DiscriminativeMotifs.PATIENT_CLASS].isin(labels)].copy()
    local = local.loc[~local.index.duplicated(keep="first")]
    if len(local[DiscriminativeMotifs.PATIENT_CLASS].unique()) != 2:
        raise ValueError("Both configured labels must be present in patient_class_df.")
    return local


def _filter_motifs(motifs_df: pd.DataFrame, p_value_threshold: float | None) -> pd.DataFrame:
    if p_value_threshold is None:
        return motifs_df.copy()
    if "p_value" not in motifs_df.columns:
        raise ValueError("motifs_df must include p_value when fanmod_p_value_threshold is not None.")
    return motifs_df[motifs_df["p_value"] < p_value_threshold].copy()


def _score_matrix(
    matrix: pd.DataFrame,
    patient_classes: pd.Series,
    config: SoftMotifSelectionConfig,
) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()

    classes = patient_classes.reindex(matrix.index)
    group_0_label, group_1_label = config.labels
    y = (classes == group_1_label).astype(int).to_numpy()
    group_0_mask = classes == group_0_label
    group_1_mask = classes == group_1_label

    rows = []
    pvalues = []
    for motif_id in matrix.columns:
        values = matrix[motif_id].astype(float).to_numpy()
        group_0_values = matrix.loc[group_0_mask, motif_id].astype(float).to_numpy()
        group_1_values = matrix.loc[group_1_mask, motif_id].astype(float).to_numpy()

        auc = _safe_auc(y, values)
        effect_score = abs(auc - 0.5) * 2.0
        mean_group_0 = float(np.mean(group_0_values))
        mean_group_1 = float(np.mean(group_1_values))
        logfc = float(np.log2((mean_group_1 + config.pseudocount) / (mean_group_0 + config.pseudocount)))
        pvalue = _safe_mannwhitney(group_0_values, group_1_values)
        transformed = np.log1p(values)
        transformed_mean = float(np.mean(transformed))
        dispersion = float(np.std(transformed) / (transformed_mean + config.epsilon))

        pvalues.append(pvalue)
        rows.append(
            {
                "motif_id": motif_id,
                "auc": auc,
                "effect_score": effect_score,
                "mean_group_0": mean_group_0,
                "mean_group_1": mean_group_1,
                "logfc_group1_vs_group0": logfc,
                "abs_logfc": abs(logfc),
                "direction": group_1_label if logfc > 0 else group_0_label if logfc < 0 else "tie",
                "mean_abundance": float(np.mean(values)),
                "abundance_raw": float(np.log1p(np.mean(values))),
                "overall_prevalence": float(np.mean(values > 0)),
                "prevalence_group_0": float(np.mean(group_0_values > 0)),
                "prevalence_group_1": float(np.mean(group_1_values > 0)),
                "specificity": float(abs(np.mean(group_1_values > 0) - np.mean(group_0_values > 0))),
                "p_value": pvalue,
                "dispersion": dispersion,
            }
        )

    scores = pd.DataFrame(rows).set_index("motif_id")
    if scores.empty:
        return scores

    qvalues = _benjamini_hochberg(pvalues) if pvalues else []
    scores["q_value"] = qvalues
    scores["confidence_score"] = _normalize_01(1.0 - scores["q_value"])
    scores["abundance_score"] = _normalize_01(scores["abundance_raw"])
    scores["prevalence_score"] = scores["overall_prevalence"].clip(0.0, 1.0)
    scores["dispersion_desirability_score"] = 1.0 - _normalize_01(
        scores["dispersion"], constant_positive_value=0.0
    )

    weights = config.weights.as_dict()
    scores["additive_score"] = (
        weights["effect"] * scores["effect_score"]
        + weights["abundance"] * scores["abundance_score"]
        + weights["prevalence"] * scores["prevalence_score"]
        + weights["confidence"] * scores["confidence_score"]
        + weights["dispersion"] * scores["dispersion_desirability_score"]
    ) / config.weights.total()
    return scores


def _safe_auc(y: np.ndarray, values: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(values)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y, values))
    except ValueError:
        return 0.5


def _safe_mannwhitney(group_0_values: np.ndarray, group_1_values: np.ndarray) -> float:
    try:
        return float(mannwhitneyu(group_0_values, group_1_values, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def _normalize_01(values: pd.Series | np.ndarray, constant_positive_value: float = 1.0) -> pd.Series:
    series = pd.Series(values, dtype=float)
    finite = series.replace([np.inf, -np.inf], np.nan)
    min_value = finite.min()
    max_value = finite.max()
    if pd.isna(min_value) or pd.isna(max_value):
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    if np.isclose(max_value, min_value):
        fill = constant_positive_value if max_value > 0 else 0.0
        return pd.Series(np.full(len(series), fill), index=series.index, dtype=float)
    return ((finite - min_value) / (max_value - min_value)).fillna(0.0).clip(0.0, 1.0)


def _benjamini_hochberg(pvalues: list[float]) -> np.ndarray:
    pvalues_array = np.asarray(pvalues, dtype=float)
    n_values = len(pvalues_array)
    order = np.argsort(pvalues_array)
    ordered = pvalues_array[order]
    adjusted = np.empty(n_values, dtype=float)
    running_min = 1.0

    for rank in range(n_values, 0, -1):
        candidate = ordered[rank - 1] * n_values / rank
        running_min = min(running_min, candidate)
        adjusted[order[rank - 1]] = running_min

    return np.clip(adjusted, 0.0, 1.0)
