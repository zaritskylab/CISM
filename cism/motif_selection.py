from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tqdm.auto import tqdm

from cism.cism import AnalyzeMotifsResult, DiscriminativeMotifs
from cism.optimization import OptunaTuningResult


@dataclass(frozen=True)
class MotifSelectionWeights:
    """Weights for the additive part of multi-objective motif scoring.

    The final additive score is divided by the sum of these weights, so only
    their relative scale matters. Use ``optimize_soft_motif_weights`` to tune
    these values against LOOCV Random Forest performance.
    """

    effect: float = 0.2
    abundance: float = 0.2
    prevalence: float = 0.2
    confidence: float = 0.2
    dispersion: float = 0.2

    def __post_init__(self) -> None:
        values = self.as_dict()
        invalid = {name: value for name, value in values.items() if value < 0.0 or value > 1.0}
        if invalid:
            raise ValueError(f"Motif selection weights must be between 0 and 1: {invalid}")
        total = sum(values.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"Motif selection weights must sum to 1.0, got {total:.6g}.")

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
    """Configuration for multi-objective soft motif selection.

    Parameters
    ----------
    labels:
        The two disease-state labels to compare. The first label is treated as
        the positive class when reporting ROC AUC from downstream CISM
        evaluation helpers.
    top_k:
        Number of motif IDs to select after ranking by final score.
    weights:
        Relative coefficients for the additive score components.
    gate:
        Stability gate applied multiplicatively after additive scoring.
    fanmod_p_value_threshold:
        Optional FANMOD p-value filter. Set to ``None`` to score all motif rows.
    final_score_cutoff:
        Optional minimum final score required before selecting motifs. Motifs
        are still capped by ``top_k`` after this cutoff is applied.
    pseudocount:
        Small positive value used in log fold-change calculations.
    epsilon:
        Small positive value used in dispersion calculations.
    """

    labels: list[str]
    top_k: int
    weights: MotifSelectionWeights = field(default_factory=MotifSelectionWeights)
    gate: StabilityGateConfig = field(default_factory=StabilityGateConfig)
    fanmod_p_value_threshold: float | None = 0.05
    final_score_cutoff: float | None = None
    pseudocount: float = 1e-9
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if len(self.labels) != 2:
            raise ValueError("Soft motif selection currently supports exactly two labels.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.final_score_cutoff is not None and self.final_score_cutoff < 0:
            raise ValueError("final_score_cutoff must be non-negative when provided.")
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
        """Return an ``InferenceFC`` that reuses the selected motif IDs.

        ``InferenceFC`` does not rediscover motifs inside each validation fold.
        It instructs CISM to train/evaluate using the already selected motif
        IDs, which is useful after a final selection pass or for deployment on
        a fixed motif panel.
        """
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


@dataclass
class SoftMotifWeightTuningResult(OptunaTuningResult):
    """Optuna result plus a ready-to-use soft motif selection config."""

    base_config: SoftMotifSelectionConfig

    @property
    def best_weights(self) -> MotifSelectionWeights:
        params = _normalize_weight_params(self.best_params)
        return MotifSelectionWeights(
            effect=params["effect"],
            abundance=params["abundance"],
            prevalence=params["prevalence"],
            confidence=params["confidence"],
            dispersion=params["dispersion"],
        )

    @property
    def best_config(self) -> SoftMotifSelectionConfig:
        params = self.best_params
        gate = self.base_config.gate
        if "tau" in params or "gamma" in params:
            gate = StabilityGateConfig(
                tau=float(params.get("tau", gate.tau)),
                gamma=float(params.get("gamma", gate.gamma)),
            )
        return SoftMotifSelectionConfig(
            labels=list(self.base_config.labels),
            top_k=self.base_config.top_k,
            weights=self.best_weights,
            gate=gate,
            fanmod_p_value_threshold=self.base_config.fanmod_p_value_threshold,
            final_score_cutoff=self.base_config.final_score_cutoff,
            pseudocount=self.base_config.pseudocount,
            epsilon=self.base_config.epsilon,
        )


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
    selected = _select_scored_motif_ids(scores, top_k=config.top_k, final_score_cutoff=config.final_score_cutoff)

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


def optimize_soft_motif_weights(
    discriminator,
    base_config: SoftMotifSelectionConfig,
    n_trials: int = 30,
    metric: str = "roc_auc",
    random_state: int = 0,
    weight_range: tuple[float, float] = (0.0, 1.0),
    n_jobs: int = 1,
    show_progress_bar: bool = True,
    storage: str | None = None,
    study_name: str | None = None,
    load_if_exists: bool = True,
    optimize_gate: bool = False,
    tau_range: tuple[float, float] = (0.6, 1.0),
    gamma_range: tuple[float, float] = (2.0, 10.0),
) -> SoftMotifWeightTuningResult:
    """Tune additive soft-motif weights with Optuna.

    Each trial samples five raw additive weight values, normalizes them to sum
    to 1.0, evaluates motif selection in
    the same leakage-safe LOOCV Random Forest framework used by
    ``evaluate_soft_motif_selection_loocv``, and maximizes the requested metric.
    The stability gate and all non-weight settings are inherited from
    ``base_config`` unless ``optimize_gate=True``.

    Pass ``storage`` and ``study_name`` to persist a study, and keep
    ``load_if_exists=True`` to resume it later with additional trials.

    Supported metrics are ``"roc_auc"``, ``"accuracy"``, and
    ``"mean_feature_count"``. The returned object exposes ``best_weights`` and
    ``best_config`` for direct reuse in ``score_soft_motifs_from_discriminator``.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for optimize_soft_motif_weights(). Install it with 'pip install optuna'."
        ) from exc

    low, high = float(weight_range[0]), float(weight_range[1])
    if low < 0 or high <= low:
        raise ValueError("weight_range must be a non-negative increasing (low, high) tuple.")
    if metric not in {"roc_auc", "accuracy", "mean_feature_count"}:
        raise ValueError("metric must be one of: roc_auc, accuracy, mean_feature_count.")
    tau_low, tau_high = float(tau_range[0]), float(tau_range[1])
    gamma_low, gamma_high = float(gamma_range[0]), float(gamma_range[1])
    if optimize_gate and (tau_low <= 0 or tau_high <= tau_low):
        raise ValueError("tau_range must be a positive increasing (low, high) tuple.")
    if optimize_gate and (gamma_low <= 0 or gamma_high <= gamma_low):
        raise ValueError("gamma_range must be a positive increasing (low, high) tuple.")

    study_exists = False
    if storage is not None and study_name is not None and load_if_exists:
        study_exists = any(
            summary.study_name == study_name
            for summary in optuna.get_all_study_summaries(storage=storage)
        )

    sampler_seed = None if study_exists else random_state
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)

    def objective(trial):
        raw_params = {
            "effect": trial.suggest_float("effect", low, high),
            "abundance": trial.suggest_float("abundance", low, high),
            "prevalence": trial.suggest_float("prevalence", low, high),
            "confidence": trial.suggest_float("confidence", low, high),
            "dispersion": trial.suggest_float("dispersion", low, high),
        }
        normalized_params = _normalize_weight_params(raw_params)
        for name, value in normalized_params.items():
            trial.set_user_attr(f"weight_{name}", value)

        weights = MotifSelectionWeights(
            effect=normalized_params["effect"],
            abundance=normalized_params["abundance"],
            prevalence=normalized_params["prevalence"],
            confidence=normalized_params["confidence"],
            dispersion=normalized_params["dispersion"],
        )
        gate = base_config.gate
        if optimize_gate:
            gate = StabilityGateConfig(
                tau=trial.suggest_float("tau", tau_low, tau_high),
                gamma=trial.suggest_float("gamma", gamma_low, gamma_high),
            )
        try:
            tuned_config = SoftMotifSelectionConfig(
                labels=list(base_config.labels),
                top_k=base_config.top_k,
                weights=weights,
                gate=gate,
                fanmod_p_value_threshold=base_config.fanmod_p_value_threshold,
                final_score_cutoff=base_config.final_score_cutoff,
                pseudocount=base_config.pseudocount,
                epsilon=base_config.epsilon,
            )
        except ValueError:
            return 0.0

        try:
            result = evaluate_soft_motif_selection_loocv(
                discriminator=discriminator,
                config=tuned_config,
                random_state=random_state,
                show_progress=False,
            )
        except ValueError as exc:
            if "No motifs selected" in str(exc) or "At least one motif selection weight" in str(exc):
                return 0.0
            raise

        if metric == "roc_auc":
            return result.get_roc_auc_score()
        if metric == "accuracy":
            metrics = result.get_metrics()
            if "accuracy" in metrics:
                return float(metrics["accuracy"])
            if "acc" in metrics:
                return float(metrics["acc"])
            return float((result.results["TP"].sum() + result.results["TN"].sum()) / result.results.shape[0])
        return float(result.results["cFeatures"].mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=load_if_exists,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=show_progress_bar)
    return SoftMotifWeightTuningResult(study=study, metric_name=metric, base_config=base_config)


def _normalize_weight_params(params: dict[str, float]) -> dict[str, float]:
    components = ["effect", "abundance", "prevalence", "confidence", "dispersion"]
    values = np.asarray([float(params[name]) for name in components], dtype=float)
    if np.any(values < 0):
        raise ValueError("Raw motif selection weight parameters must be non-negative.")
    total = float(values.sum())
    if total <= 0:
        return {name: 1.0 / len(components) for name in components}
    normalized = values / total
    return {name: float(value) for name, value in zip(components, normalized)}


def compute_loocv_selection_stability(
    motifs_df: pd.DataFrame,
    patient_class_df: pd.DataFrame,
    config: SoftMotifSelectionConfig,
    motif_col: str = "ID",
    show_progress: bool = True,
) -> pd.Series:
    """Estimate motif stability as training-fold top-k selection frequency under LOOCV."""
    local_classes = _prepare_patient_classes(patient_class_df, config.labels)
    selected_counts: defaultdict[Any, int] = defaultdict(int)
    eligible_counts: defaultdict[Any, int] = defaultdict(int)

    for held_out_patient in tqdm(
        local_classes.index,
        desc="Estimating motif stability",
        leave=False,
        disable=not show_progress,
    ):
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
    show_progress: bool = True,
) -> SoftMotifLOOCVResult:
    """Evaluate soft motif selection in a leakage-safe leave-one-patient-out loop."""
    patient_class_df = discriminator.get_patients_class(config.labels)
    local_classes = _prepare_patient_classes(patient_class_df, config.labels)
    full_motifs_df = discriminator.cism.motifs_dataset
    analyze_rows = []
    fold_score_tables = {}
    selected_by_patient = {}

    for held_out_patient in tqdm(
        local_classes.index,
        desc="Evaluating soft motif LOOCV",
        leave=False,
        disable=not show_progress,
    ):
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


def evaluate_soft_motif_score_cutoff_sweep_loocv(
    discriminator,
    config: SoftMotifSelectionConfig,
    score_cutoffs: list[float] | np.ndarray,
    random_state: int = 0,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Sweep final-score cutoffs in leakage-safe LOOCV.

    Motif scores are computed using only each training fold. For every cutoff,
    motifs with ``final_score >= cutoff`` are selected up to ``config.top_k``;
    a Random Forest is fit on those selected motifs and predicts the held-out
    patient. Folds with no selected motifs are counted and excluded from model
    metrics for that cutoff.
    """
    cutoffs = sorted({float(cutoff) for cutoff in score_cutoffs})
    if any(cutoff < 0 for cutoff in cutoffs):
        raise ValueError("score_cutoffs must be non-negative.")

    patient_class_df = discriminator.get_patients_class(config.labels)
    local_classes = _prepare_patient_classes(patient_class_df, config.labels)
    full_motifs_df = discriminator.cism.motifs_dataset
    scoring_config = replace(config, final_score_cutoff=None)
    rows_by_cutoff = {
        cutoff: {
            "true": [],
            "pred": [],
            "positive_probability": [],
            "selected_counts": [],
            "empty_folds": 0,
        }
        for cutoff in cutoffs
    }

    for held_out_patient in tqdm(
        local_classes.index,
        desc="Sweeping soft motif score cutoffs",
        leave=False,
        disable=not show_progress,
    ):
        train_classes = local_classes.drop(index=held_out_patient)
        test_classes = local_classes.loc[[held_out_patient]]
        train_motifs = full_motifs_df[full_motifs_df["Patient_uId"].isin(train_classes.index)]
        test_motifs = full_motifs_df[full_motifs_df["Patient_uId"].isin(test_classes.index)]

        train_result = score_soft_motifs(
            motifs_df=train_motifs,
            patient_class_df=train_classes,
            config=scoring_config,
            stability=None,
        )
        y_train = train_classes[DiscriminativeMotifs.PATIENT_CLASS]
        y_test = test_classes[DiscriminativeMotifs.PATIENT_CLASS].iloc[0]
        test_matrix = build_patient_motif_matrix(test_motifs, patient_ids=test_classes.index)

        for cutoff in cutoffs:
            selected = _select_from_score_table(
                train_result.scores,
                motif_col="ID",
                top_k=config.top_k,
                final_score_cutoff=cutoff,
            )
            cutoff_rows = rows_by_cutoff[cutoff]
            cutoff_rows["selected_counts"].append(len(selected))
            if not selected:
                cutoff_rows["empty_folds"] += 1
                continue

            x_train = train_result.patient_motif_matrix.reindex(columns=selected, fill_value=0.0)
            x_test = test_matrix.reindex(columns=selected, fill_value=0.0)
            clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
            clf.fit(x_train, y_train)
            pred = clf.predict(x_test)[0]
            prob = clf.predict_proba(x_test)
            positive_index = np.where(clf.classes_ == config.labels[0])[0][0]

            cutoff_rows["true"].append(y_test)
            cutoff_rows["pred"].append(pred)
            cutoff_rows["positive_probability"].append(float(prob[0][positive_index]))

    summary_rows = []
    for cutoff in cutoffs:
        cutoff_rows = rows_by_cutoff[cutoff]
        true = cutoff_rows["true"]
        pred = cutoff_rows["pred"]
        probabilities = cutoff_rows["positive_probability"]
        selected_counts = cutoff_rows["selected_counts"]
        valid_folds = len(true)

        roc_auc = np.nan
        accuracy = np.nan
        balanced_accuracy = np.nan
        if valid_folds > 0:
            accuracy = float(accuracy_score(true, pred))
            if len(set(true)) == 2:
                balanced_accuracy = float(balanced_accuracy_score(true, pred))
                binary_true = [1 if label == config.labels[0] else 0 for label in true]
                roc_auc = float(roc_auc_score(binary_true, probabilities))

        summary_rows.append(
            {
                "final_score_cutoff": cutoff,
                "roc_auc": roc_auc,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "mean_selected_motifs": float(np.mean(selected_counts)) if selected_counts else np.nan,
                "median_selected_motifs": float(np.median(selected_counts)) if selected_counts else np.nan,
                "min_selected_motifs": int(np.min(selected_counts)) if selected_counts else 0,
                "max_selected_motifs": int(np.max(selected_counts)) if selected_counts else 0,
                "valid_folds": valid_folds,
                "empty_folds": int(cutoff_rows["empty_folds"]),
                "total_folds": len(local_classes.index),
            }
        )

    return pd.DataFrame(summary_rows)


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


def _select_scored_motif_ids(
    scores: pd.DataFrame,
    top_k: int,
    final_score_cutoff: float | None,
) -> list[Any]:
    selected_scores = scores
    if final_score_cutoff is not None:
        selected_scores = selected_scores[selected_scores["final_score"] >= final_score_cutoff]
    return selected_scores.head(top_k).index.tolist()


def _select_from_score_table(
    scores: pd.DataFrame,
    motif_col: str,
    top_k: int,
    final_score_cutoff: float | None,
) -> list[Any]:
    selected_scores = scores
    if final_score_cutoff is not None:
        selected_scores = selected_scores[selected_scores["final_score"] >= final_score_cutoff]
    return selected_scores.head(top_k)[motif_col].tolist()


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
