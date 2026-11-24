"""Leakage-aware grouped experiments for the Gold exercise datasets."""

from __future__ import annotations

import json
import logging
import math
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.features.biomechanics import feature_dictionary

LOGGER = logging.getLogger(__name__)
EXERCISES = ("plank", "hollow_body")
SUMMARY_STATISTICS = ("mean", "std", "min", "max", "range", "slope")
TARGET_COLUMN = "label"
GROUP_COLUMN = "participant_id"
POSITIVE_LABEL = "incorrect"
SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
SEARCH_ITERATIONS = 6
MIN_GROUPS_FOR_STABLE_SELECTION = 12
FEATURE_METADATA_EXCLUSIONS = {
    "video_id",
    "participant_id",
    "exercise",
    "window_id",
    "label",
    "error_type",
    "camera_view",
    "source_video",
    "window_start_ms",
    "window_end_ms",
    "frame_ids",
    "n_sampled_frames",
    "n_valid_pose_frames",
}


def _feature_columns(frame: pd.DataFrame, exercise: str) -> list[str]:
    dictionary = feature_dictionary(exercise)
    prefixes = tuple(dictionary["features"])
    columns = [
        f"{prefix}_{statistic}"
        for prefix in prefixes
        for statistic in SUMMARY_STATISTICS
        if f"{prefix}_{statistic}" in frame.columns
    ]
    if not columns:
        raise ValueError(f"No se encontraron características Gold para {exercise}.")
    forbidden = FEATURE_METADATA_EXCLUSIONS.intersection(columns)
    if forbidden:
        raise ValueError(f"Se intentó usar metadata como característica: {sorted(forbidden)}")
    return columns


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Adapt classifiers that require integer targets to string class labels."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> LabelEncodedClassifier:
        self.classes_, encoded_labels = np.unique(np.asarray(y), return_inverse=True)
        self.estimator_ = clone(self.estimator).fit(X, encoded_labels)
        return self

    def predict(self, X: Any) -> np.ndarray:
        encoded_predictions = np.asarray(self.estimator_.predict(X), dtype=int)
        return self.classes_[encoded_predictions]

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.estimator_.feature_importances_


def load_exercise_dataset(
    project_root: Path,
    exercise: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], pd.DataFrame]:
    """Load target, participant groups, numeric predictors, and untouched metadata."""
    exercise = exercise.lower()
    if exercise not in EXERCISES:
        raise ValueError(f"Ejercicio no soportado: {exercise}")
    path = project_root / "data" / "gold" / f"{exercise}_features.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset Gold: {path}")
    frame = pd.read_parquet(path, engine="pyarrow")
    required = {TARGET_COLUMN, GROUP_COLUMN, "video_id", "exercise"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {path}: {sorted(missing)}")
    if frame.empty:
        raise ValueError(
            f"El dataset {path} no tiene ventanas; no se puede entrenar este ejercicio."
        )
    if frame["exercise"].nunique(dropna=False) != 1 or frame["exercise"].iloc[0] != exercise:
        raise ValueError(f"El dataset contiene un ejercicio distinto a {exercise}.")
    labels = set(frame[TARGET_COLUMN].dropna().astype(str))
    if labels != {"correct", "incorrect"} or frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"Se esperaba target binario correct/incorrect; se encontró {labels}.")
    if frame[GROUP_COLUMN].isna().any():
        raise ValueError("participant_id contiene valores nulos; no se puede agrupar el CV.")

    columns = _feature_columns(frame, exercise)
    predictors = frame[columns].apply(pd.to_numeric, errors="coerce")
    predictors = predictors.replace([np.inf, -np.inf], np.nan)
    metadata = frame.drop(columns=columns).copy()
    return (
        predictors,
        frame[TARGET_COLUMN].astype(str),
        frame[GROUP_COLUMN].astype(str),
        columns,
        metadata,
    )


def _validate_splits(
    frame: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if groups.nunique() < n_splits:
        raise ValueError(
            f"Se requieren {n_splits} grupos, pero solo hay {groups.nunique()} participantes."
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(frame, labels, groups))
    for fold_number, (train_indices, valid_indices) in enumerate(splits, start=1):
        train_labels = set(labels.iloc[train_indices])
        valid_labels = set(labels.iloc[valid_indices])
        if len(train_labels) != 2 or len(valid_labels) != 2:
            raise ValueError(
                f"El fold {fold_number} no contiene ambas clases en train y validación: "
                f"train={train_labels}, valid={valid_labels}."
            )
        overlap = set(groups.iloc[train_indices]) & set(groups.iloc[valid_indices])
        if overlap:
            raise RuntimeError(f"Data leakage en fold {fold_number}: {sorted(overlap)}")
    return splits


def _audit_dataset(
    exercise: str,
    predictors: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    feature_columns: list[str],
    metadata: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    report_directory: Path,
) -> dict[str, Any]:
    report_directory.mkdir(parents=True, exist_ok=True)
    metadata_columns = [
        name
        for name in (
            "video_id",
            "participant_id",
            "exercise",
            "window_id",
            "window_start_ms",
            "window_end_ms",
            "label",
            "error_type",
            "camera_view",
            "source_video",
        )
        if name in metadata.columns
    ]
    duplicates = int(
        metadata.duplicated(
            [column for column in ("video_id", "window_start_ms", "window_end_ms") if column in metadata]
        ).sum()
    )
    summary = {
        "exercise": exercise,
        "samples": int(len(labels)),
        "participants": int(groups.nunique()),
        "videos": int(metadata["video_id"].nunique()),
        "windows": int(len(labels)),
        "predictor_count": int(len(feature_columns)),
        "predictors": feature_columns,
        "class_counts": labels.value_counts().to_dict(),
        "participants_per_class": metadata.assign(_group=groups, _label=labels).groupby("_label")[
            "_group"
        ].nunique().to_dict(),
        "samples_per_participant": groups.value_counts().sort_index().to_dict(),
        "samples_per_video": metadata["video_id"].value_counts().sort_index().to_dict(),
        "error_type_counts": (
            metadata["error_type"].fillna("<null>").value_counts().to_dict()
            if "error_type" in metadata
            else {}
        ),
        "missing_values_per_feature": predictors.isna().sum().astype(int).to_dict(),
        "infinite_values_per_feature": {
            column: int(np.isinf(predictors[column].dropna().to_numpy(dtype=float)).sum())
            for column in predictors
        },
        "constant_features": [
            column
            for column in predictors
            if predictors[column].nunique(dropna=True) <= 1
        ],
        "duplicate_video_window_keys": duplicates,
        "all_scalar_duplicate_rows": int(
            metadata.drop(columns=["frame_ids"], errors="ignore").duplicated().sum()
        ),
        "metadata_columns": metadata_columns,
        "label_only_participants": {
            str(participant): labels_for_participant.index.tolist()
            for participant, labels_for_participant in (
                metadata.assign(_group=groups, _label=labels)
                .groupby("_group")["_label"]
                .apply(lambda values: values.value_counts())
                .groupby(level=0)
            )
            if len(labels_for_participant.index) == 1
        },
        "outer_cv": {
            "strategy": "StratifiedGroupKFold",
            "n_splits": len(splits),
            "seed": SEED,
            "folds": [
                {
                    "fold": index,
                    "train_samples": int(len(train)),
                    "valid_samples": int(len(valid)),
                    "train_participants": int(groups.iloc[train].nunique()),
                    "valid_participants": int(groups.iloc[valid].nunique()),
                    "valid_classes": labels.iloc[valid].value_counts().to_dict(),
                    "participant_overlap": [],
                }
                for index, (train, valid) in enumerate(splits, start=1)
            ],
        },
    }
    (report_directory / "dataset_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictors.describe().T.to_csv(report_directory / "feature_statistics.csv")
    predictors.isna().sum().rename("missing_count").to_csv(
        report_directory / "feature_missing_values.csv"
    )
    labels.value_counts().rename_axis("label").to_csv(report_directory / "class_counts.csv")
    metadata.assign(participant_id=groups, label=labels).groupby(
        ["participant_id", "label"]
    ).size().unstack(fill_value=0).to_csv(report_directory / "participant_class_counts.csv")
    metadata.assign(video_id=metadata["video_id"]).groupby(
        ["video_id", "label"]
    ).size().unstack(fill_value=0).to_csv(report_directory / "video_class_counts.csv")

    class_counts = labels.value_counts().reindex(["correct", "incorrect"], fill_value=0)
    figure, axis = plt.subplots(figsize=(6, 4))
    class_counts.plot.bar(ax=axis, color=["#4C9F70", "#D95D39"])
    axis.set(title=f"{exercise}: distribución de clases", xlabel="Etiqueta", ylabel="Ventanas")
    figure.tight_layout()
    figure.savefig(report_directory / "class_distribution.png", dpi=150)
    plt.close(figure)

    correlation = predictors.corr().fillna(0.0)
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(feature_columns)), feature_columns, rotation=90, fontsize=6)
    axis.set_yticks(range(len(feature_columns)), feature_columns, fontsize=6)
    figure.colorbar(image, ax=axis, label="Correlación de Pearson")
    axis.set_title(f"{exercise}: correlación entre características")
    figure.tight_layout()
    figure.savefig(report_directory / "feature_correlations.png", dpi=180)
    plt.close(figure)
    return summary


def _candidate_models(exercise: str) -> dict[str, tuple[Pipeline, dict[str, list[Any]], bool]]:
    seed = SEED

    def pipeline(classifier: Any, *, scale: bool) -> Pipeline:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", classifier))
        return Pipeline(steps)

    return {
        "logistic_regression": (
            pipeline(LogisticRegression(max_iter=3000, random_state=seed), scale=True),
            {
                "classifier__C": [0.05, 0.2, 1.0, 5.0, 20.0],
                "classifier__class_weight": [None, "balanced"],
            },
            True,
        ),
        "svm_rbf": (
            pipeline(SVC(random_state=seed), scale=True),
            {
                "classifier__C": [0.1, 0.5, 1.0, 5.0, 20.0],
                "classifier__gamma": ["scale", "auto", 0.01, 0.1],
                "classifier__class_weight": [None, "balanced"],
            },
            True,
        ),
        "random_forest": (
            pipeline(
                RandomForestClassifier(random_state=seed, n_jobs=1),
                scale=False,
            ),
            {
                "classifier__n_estimators": [100, 250, 400],
                "classifier__max_depth": [None, 3, 5],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", 0.7, 1.0],
                "classifier__class_weight": [None, "balanced"],
            },
            False,
        ),
        "extra_trees": (
            pipeline(
                ExtraTreesClassifier(random_state=seed, n_jobs=1),
                scale=False,
            ),
            {
                "classifier__n_estimators": [100, 250, 400],
                "classifier__max_depth": [None, 3, 5],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", 0.7, 1.0],
                "classifier__class_weight": [None, "balanced"],
            },
            False,
        ),
        "hist_gradient_boosting": (
            pipeline(
                HistGradientBoostingClassifier(random_state=seed),
                scale=False,
            ),
            {
                "classifier__max_iter": [50, 100, 200],
                "classifier__learning_rate": [0.03, 0.08, 0.15],
                "classifier__max_leaf_nodes": [3, 7, 15],
                "classifier__l2_regularization": [0.0, 0.5, 2.0],
                "classifier__class_weight": [None, "balanced"],
            },
            False,
        ),
        "xgboost": (
            pipeline(
                LabelEncodedClassifier(
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=seed,
                        n_jobs=1,
                    )
                ),
                scale=False,
            ),
            {
                "classifier__estimator__n_estimators": [50, 100, 200],
                "classifier__estimator__max_depth": [2, 3, 4],
                "classifier__estimator__learning_rate": [0.03, 0.1, 0.2],
                "classifier__estimator__subsample": [0.7, 1.0],
                "classifier__estimator__colsample_bytree": [0.7, 1.0],
                "classifier__estimator__reg_lambda": [1.0, 5.0, 10.0],
            },
            False,
        ),
    }


def _positive_scores(estimator: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classifier = estimator.named_steps["classifier"]
    classes = list(classifier.classes_)
    positive_index = classes.index(POSITIVE_LABEL)
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(features))[:, positive_index]
    decision = np.asarray(estimator.decision_function(features))
    return decision if positive_index == 1 else -decision


def _metric_row(
    y_true: pd.Series,
    y_pred: np.ndarray,
    scores: np.ndarray,
    *,
    exercise: str,
    model: str,
    fold: int,
) -> dict[str, Any]:
    both_classes = y_true.nunique() == 2
    return {
        "exercise": exercise,
        "model": model,
        "fold": fold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_incorrect": float(
            precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "recall_incorrect": float(
            recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "f1_incorrect": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true == POSITIVE_LABEL, scores)) if both_classes else math.nan,
        "average_precision": float(
            average_precision_score(y_true == POSITIVE_LABEL, scores)
        ) if both_classes else math.nan,
    }


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number])
        if column != "fold"
    ]
    result = metrics.groupby(["exercise", "model"])[numeric].agg(["mean", "std"])
    result.columns = [f"{metric}_{statistic}" for metric, statistic in result.columns]
    return result.reset_index()


def _nested_search(
    estimator: Pipeline,
    distributions: dict[str, list[Any]],
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    exercise: str,
    model_name: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows = []
    out_of_fold_predictions = np.empty(len(labels), dtype=object)
    out_of_fold_scores = np.full(len(labels), np.nan, dtype=float)
    search_records = []
    for fold_number, (train_indices, valid_indices) in enumerate(outer_splits, start=1):
        train_x = features.iloc[train_indices]
        train_y = labels.iloc[train_indices]
        train_groups = groups.iloc[train_indices]
        inner_folds = min(INNER_FOLDS, train_groups.nunique())
        inner_splitter = StratifiedGroupKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=SEED + fold_number,
        )
        inner_splits = list(inner_splitter.split(train_x, train_y, train_groups))
        for inner_number, (inner_train, inner_valid) in enumerate(inner_splits, start=1):
            overlap = set(train_groups.iloc[inner_train]) & set(train_groups.iloc[inner_valid])
            if overlap:
                raise RuntimeError(
                    f"Data leakage en CV interno {fold_number}/{inner_number}: {sorted(overlap)}"
                )
            if train_y.iloc[inner_valid].nunique() != 2:
                raise ValueError(
                    f"El fold interno {fold_number}/{inner_number} no tiene ambas clases."
                )
        search = RandomizedSearchCV(
            estimator=clone(estimator),
            param_distributions=distributions,
            n_iter=SEARCH_ITERATIONS,
            scoring="f1_macro",
            cv=inner_splits,
            refit=True,
            random_state=SEED + fold_number,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(train_x, train_y)
        valid_x = features.iloc[valid_indices]
        valid_y = labels.iloc[valid_indices]
        predictions = search.predict(valid_x)
        scores = _positive_scores(search.best_estimator_, valid_x)
        out_of_fold_predictions[valid_indices] = predictions
        out_of_fold_scores[valid_indices] = scores
        rows.append(
            _metric_row(
                valid_y,
                predictions,
                scores,
                exercise=exercise,
                model=model_name,
                fold=fold_number,
            )
        )
        search_records.append(
            {
                "fold": fold_number,
                "best_params": search.best_params_,
                "inner_best_f1_macro": float(search.best_score_),
                "train_participants": int(train_groups.nunique()),
                "validation_participants": sorted(
                    groups.iloc[valid_indices].unique().tolist()
                ),
            }
        )
    if np.isnan(out_of_fold_scores).any() or any(
        prediction is None for prediction in out_of_fold_predictions
    ):
        raise RuntimeError("No se produjeron predicciones OOF para todas las ventanas.")
    return pd.DataFrame(rows), out_of_fold_predictions, out_of_fold_scores, search_records


def _plot_model_comparison(summary: pd.DataFrame, output_path: Path, exercise: str) -> None:
    metric = "f1_macro_mean"
    error = "f1_macro_std"
    ordered = summary.sort_values(metric, ascending=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        ordered["model"],
        ordered[metric],
        yerr=ordered[error].fillna(0),
        capsize=4,
        color="#457B9D",
    )
    axis.set(
        title=f"{exercise}: F1 macro CV por participante (media ± SD)",
        xlabel="Modelo",
        ylabel="F1 macro",
        ylim=(0, 1),
    )
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_confusion_and_curves(
    labels: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray,
    report_directory: Path,
    exercise: str,
) -> None:
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    class_order = ["correct", "incorrect"]
    confusion = confusion_matrix(labels, predictions, labels=class_order)
    pd.DataFrame(
        confusion,
        index=[f"actual_{label}" for label in class_order],
        columns=[f"predicted_{label}" for label in class_order],
    ).to_csv(report_directory / "confusion_matrix.csv")
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(range(2), class_order)
    axis.set_yticks(range(2), class_order)
    axis.set(xlabel="Predicción", ylabel="Real", title=f"{exercise}: matriz de confusión OOF")
    for row_index in range(2):
        for column_index in range(2):
            axis.text(column_index, row_index, str(confusion[row_index, column_index]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(report_directory / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    report = classification_report(
        labels,
        predictions,
        labels=class_order,
        output_dict=True,
        zero_division=0,
    )
    (report_directory / "classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(
        labels == POSITIVE_LABEL,
        scores,
        ax=axes[0],
        name=exercise,
    )
    axes[0].set_title("ROC (predicciones externas OOF)")
    PrecisionRecallDisplay.from_predictions(
        labels == POSITIVE_LABEL,
        scores,
        ax=axes[1],
        name=exercise,
    )
    axes[1].set_title("Precision–Recall (OOF)")
    figure.tight_layout()
    figure.savefig(report_directory / "roc_pr_curves.png", dpi=150)
    plt.close(figure)


def _save_feature_importance(
    estimator: Pipeline,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    classifier = estimator.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance = np.asarray(classifier.feature_importances_, dtype=float)
        method = "tree_impurity_importance_not_causal"
    elif hasattr(classifier, "coef_"):
        importance = np.mean(np.abs(np.asarray(classifier.coef_, dtype=float)), axis=0)
        method = "mean_absolute_coefficient_not_causal"
    else:
        return
    pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance,
            "method": method,
        }
    ).sort_values("importance", ascending=False).to_csv(output_path, index=False)


def run_experiment(project_root: Path, exercise: str) -> dict[str, Any]:
    """Run EDA, shared grouped CV, nested tuning, model selection, and final fit."""
    exercise = exercise.lower()
    predictors, labels, groups, feature_columns, metadata = load_exercise_dataset(
        project_root,
        exercise,
    )
    report_directory = project_root / "reports" / exercise
    report_directory.mkdir(parents=True, exist_ok=True)
    outer_splits = _validate_splits(
        predictors,
        labels,
        groups,
        n_splits=min(OUTER_FOLDS, groups.nunique()),
        seed=SEED,
    )
    audit = _audit_dataset(
        exercise,
        predictors,
        labels,
        groups,
        feature_columns,
        metadata,
        outer_splits,
        report_directory,
    )
    candidates = _candidate_models(exercise)
    initial_rows = []
    for model_name, (estimator, _, _) in candidates.items():
        LOGGER.info("Evaluando %s para %s con folds agrupados compartidos.", model_name, exercise)
        for fold_number, (train_indices, valid_indices) in enumerate(outer_splits, start=1):
            fitted = clone(estimator).fit(
                predictors.iloc[train_indices],
                labels.iloc[train_indices],
            )
            valid_x = predictors.iloc[valid_indices]
            valid_y = labels.iloc[valid_indices]
            initial_rows.append(
                _metric_row(
                    valid_y,
                    fitted.predict(valid_x),
                    _positive_scores(fitted, valid_x),
                    exercise=exercise,
                    model=model_name,
                    fold=fold_number,
                )
            )
    initial_metrics = pd.DataFrame(initial_rows)
    initial_summary = _summarize_metrics(initial_metrics)
    initial_metrics.to_csv(report_directory / "cv_initial_fold_metrics.csv", index=False)
    initial_summary.to_csv(report_directory / "cv_initial_model_comparison.csv", index=False)
    _plot_model_comparison(
        initial_summary,
        report_directory / "cv_initial_f1_macro.png",
        exercise,
    )

    top_models = (
        initial_summary.sort_values(
            ["f1_macro_mean", "recall_incorrect_mean", "balanced_accuracy_mean"],
            ascending=False,
        )["model"]
        .head(2)
        .tolist()
    )
    nested_rows = []
    nested_predictions: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    for model_name in top_models:
        estimator, distributions, _ = candidates[model_name]
        LOGGER.info("Nested CV agrupado y tuning para %s/%s.", exercise, model_name)
        rows, predictions, scores, searches = _nested_search(
            estimator,
            distributions,
            predictors,
            labels,
            groups,
            outer_splits,
            exercise=exercise,
            model_name=model_name,
        )
        nested_rows.append(rows)
        nested_predictions[model_name] = (predictions, scores, searches)
    nested_metrics = pd.concat(nested_rows, ignore_index=True)
    nested_summary = _summarize_metrics(nested_metrics)
    nested_metrics.to_csv(report_directory / "nested_cv_fold_metrics.csv", index=False)
    nested_summary.to_csv(report_directory / "nested_cv_model_comparison.csv", index=False)
    _plot_model_comparison(
        nested_summary,
        report_directory / "nested_cv_f1_macro.png",
        exercise,
    )

    selected_model = nested_summary.sort_values(
        ["f1_macro_mean", "recall_incorrect_mean", "balanced_accuracy_mean", "f1_macro_std"],
        ascending=[False, False, False, True],
    ).iloc[0]["model"]
    selection = nested_summary.set_index("model").loc[selected_model]
    selected_predictions, selected_scores, search_records = nested_predictions[selected_model]
    _plot_confusion_and_curves(
        labels,
        selected_predictions,
        selected_scores,
        report_directory,
        exercise,
    )
    (report_directory / "nested_search_parameters.json").write_text(
        json.dumps(search_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _, distributions, requires_scaling = candidates[selected_model]
    final_search = RandomizedSearchCV(
        estimator=clone(candidates[selected_model][0]),
        param_distributions=distributions,
        n_iter=SEARCH_ITERATIONS,
        scoring="f1_macro",
        cv=outer_splits,
        refit=True,
        random_state=SEED,
        n_jobs=1,
        error_score="raise",
    )
    final_search.fit(predictors, labels)
    final_model = final_search.best_estimator_
    model_directory = project_root / "models" / exercise
    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = model_directory / "model.joblib"
    joblib.dump(final_model, model_path)
    _save_feature_importance(
        final_model,
        feature_columns,
        report_directory / "feature_importance.csv",
    )

    label_table = metadata.assign(_label=labels, _participant=groups)
    participant_label_support = label_table.groupby("_participant")["_label"].nunique()
    provisional_reasons = []
    if groups.nunique() < MIN_GROUPS_FOR_STABLE_SELECTION:
        provisional_reasons.append(
            f"solo {groups.nunique()} participantes; el umbral operativo es "
            f"{MIN_GROUPS_FOR_STABLE_SELECTION}"
        )
    if (participant_label_support == 1).any():
        provisional_reasons.append(
            f"{int((participant_label_support == 1).sum())} participantes aparecen con una sola clase"
        )
    model_metadata = {
        "exercise": exercise,
        "algorithm": selected_model,
        "best_hyperparameters": final_search.best_params_,
        "features": feature_columns,
        "target": TARGET_COLUMN,
        "class_encoding": {
            "correct": "correct",
            "incorrect": "incorrect",
            "positive_class_for_metrics": POSITIVE_LABEL,
        },
        "metrics": {
            metric.removesuffix("_mean"): float(selection[metric])
            for metric in selection.index
            if metric.endswith("_mean")
        }
        | {
            f"{metric.removesuffix('_std')}_std": (
                float(selection[metric]) if pd.notna(selection[metric]) else None
            )
            for metric in selection.index
            if metric.endswith("_std")
        },
        "cross_validation": {
            "strategy": "Nested StratifiedGroupKFold",
            "outer_folds": len(outer_splits),
            "inner_folds": min(INNER_FOLDS, groups.nunique() - 1),
            "group_column": GROUP_COLUMN,
            "seed": SEED,
            "outer_oof_metrics_are_primary": True,
        },
        "samples": int(len(labels)),
        "participants": int(groups.nunique()),
        "videos": int(metadata["video_id"].nunique()),
        "provisional_selection": bool(provisional_reasons),
        "selection_limitations": provisional_reasons,
        "python_version": platform.python_version(),
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "final_fit_group_cv_macro_f1": float(final_search.best_score_),
        "scaling_applied": requires_scaling,
        "feature_importance_warning": (
            "Las importancias son asociativas y no representan causalidad biomecánica."
        ),
        "dataset_audit": audit,
    }
    (model_directory / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    result = {
        "exercise": exercise,
        "selected_model": selected_model,
        "provisional_selection": bool(provisional_reasons),
        "selection_limitations": provisional_reasons,
        "metrics": model_metadata["metrics"],
        "models_compared": list(candidates),
        "tuned_candidates": top_models,
        "samples": len(labels),
        "participants": groups.nunique(),
        "model_path": model_path,
        "report_directory": report_directory,
    }
    LOGGER.info("Experimento completado: %s", result)
    return result


def main(exercise: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    project_root = Path(__file__).resolve().parents[2]
    run_experiment(project_root, exercise)
