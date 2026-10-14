from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, extract_query_features
from .pipeline import AdaptiveContextPipeline
from .schemas import Chunk, QuestionRecord
from .text import normalize_answer, token_f1


VALID_LABELS = {"simple", "moderate", "complex"}


def evidence_match_score(evidence: str, chunk_text: str) -> float:
    evidence_normalized = normalize_answer(evidence)
    chunk_normalized = normalize_answer(chunk_text)
    if not evidence_normalized or not chunk_normalized:
        return 0.0
    if evidence_normalized in chunk_normalized or chunk_normalized in evidence_normalized:
        shorter = min(len(evidence_normalized.split()), len(chunk_normalized.split()))
        longer = max(len(evidence_normalized.split()), len(chunk_normalized.split()))
        return min(1.0, 0.75 + 0.25 * shorter / max(longer, 1))
    return token_f1(evidence_normalized, chunk_normalized)


def map_evidence_to_chunks(
    evidence: list[str],
    chunks: list[Chunk],
    threshold: float = 0.55,
) -> set[str]:
    matched: set[str] = set()
    for span in evidence:
        scored = [(evidence_match_score(span, chunk.text), chunk.id) for chunk in chunks]
        if not scored:
            continue
        best_score, best_id = max(scored)
        if best_score >= threshold:
            matched.add(best_id)
    return matched


def derive_budget_label(
    question: QuestionRecord,
    ranked_chunk_ids: list[str],
    gold_chunk_ids: set[str],
    simple_budget: int,
    moderate_budget: int,
) -> str:
    if question.difficulty in VALID_LABELS:
        return str(question.difficulty)
    if not question.evidence:
        return "complex"
    positions = [
        ranked_chunk_ids.index(chunk_id) + 1
        for chunk_id in gold_chunk_ids
        if chunk_id in ranked_chunk_ids
    ]
    if len(positions) < len(gold_chunk_ids) or not positions:
        return "complex"
    required = max(positions)
    if required <= simple_budget:
        return "simple"
    if required <= moderate_budget:
        return "moderate"
    return "complex"


def build_budget_training_frame(
    pipeline: AdaptiveContextPipeline,
    questions: list[QuestionRecord],
    evidence_threshold: float = 0.55,
) -> pd.DataFrame:
    chunks_by_document: dict[str, list[Chunk]] = {}
    for chunk in pipeline.chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)
    adaptive = pipeline.config["adaptive"]
    rows: list[dict] = []
    for question in questions:
        candidates = pipeline.retrieve_candidates(question.question, [question.document_id])
        features = extract_query_features(question.question, candidates)
        gold = map_evidence_to_chunks(
            question.evidence,
            chunks_by_document.get(question.document_id, []),
            threshold=evidence_threshold,
        )
        label = derive_budget_label(
            question,
            [candidate.chunk.id for candidate in candidates],
            gold,
            int(adaptive["simple_budget"]),
            int(adaptive["moderate_budget"]),
        )
        rows.append(
            {
                "question_id": question.id,
                "document_id": question.document_id,
                "split": question.split,
                "label": label,
                "gold_evidence_chunks": len(gold),
                **features,
            }
        )
    return pd.DataFrame(rows)


def train_budget_controller(
    pipeline: AdaptiveContextPipeline,
    questions: list[QuestionRecord],
    output_path: str | Path,
    training_csv_path: str | Path | None = None,
) -> dict:
    threshold = float(pipeline.config["evaluation"]["evidence_match_threshold"])
    frame = build_budget_training_frame(pipeline, questions, threshold)
    if frame.empty:
        raise ValueError("No budget-controller training rows were produced.")
    if frame["label"].nunique() < 2:
        raise ValueError("Budget-controller training needs at least two difficulty classes.")
    X = frame[FEATURE_NAMES].to_numpy(dtype=float)
    y = frame["label"].to_numpy()
    seed = int(pipeline.config["project"].get("random_seed", 42))

    models = {
        "multinomial_logistic_regression": SklearnPipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
        ),
    }
    counts = frame["label"].value_counts()
    minimum_class_count = int(counts.min())
    cv_scores: dict[str, float | None] = {}
    if len(frame) >= 9 and minimum_class_count >= 2:
        folds = min(5, minimum_class_count)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=splitter, scoring="balanced_accuracy")
            cv_scores[name] = float(np.mean(scores))
        chosen_name = max(cv_scores, key=lambda name: float(cv_scores[name] or -1.0))
    else:
        chosen_name = "random_forest"
        cv_scores = {name: None for name in models}

    model = models[chosen_name]
    model.fit(X, y)
    predictions = model.predict(X)
    adaptive = pipeline.config["adaptive"]
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "label_to_budget": {
            "simple": int(adaptive["simple_budget"]),
            "moderate": int(adaptive["moderate_budget"]),
            "complex": int(adaptive["complex_budget"]),
        },
        "metadata": {
            "chosen_model": chosen_name,
            "training_rows": int(len(frame)),
            "class_distribution": {key: int(value) for key, value in counts.items()},
            "cv_balanced_accuracy": cv_scores,
            "training_balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    if training_csv_path:
        csv_path = Path(training_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)

    report = {
        **artifact["metadata"],
        "artifact_path": str(destination.resolve()),
        "classification_report_on_training_data": classification_report(
            y, predictions, output_dict=True, zero_division=0
        ),
    }
    metrics_path = destination.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

