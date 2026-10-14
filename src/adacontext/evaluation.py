from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from .pipeline import AdaptiveContextPipeline
from .schemas import Chunk, QuestionRecord
from .text import exact_match, token_f1
from .training import map_evidence_to_chunks


def best_reference_score(prediction: str, references: list[str], metric) -> float:
    return max((metric(prediction, reference) for reference in references), default=0.0)


def _bootstrap_interval(
    values: np.ndarray,
    samples: int,
    random_seed: int,
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(random_seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_strategies(
    pipeline: AdaptiveContextPipeline,
    questions: list[QuestionRecord],
    strategies: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = pipeline.config
    strategies = strategies or list(config["evaluation"]["strategies"])
    threshold = float(config["evaluation"]["evidence_match_threshold"])
    chunks_by_document: dict[str, list[Chunk]] = {}
    for chunk in pipeline.chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

    rows: list[dict] = []
    for question in questions:
        document_chunks = chunks_by_document.get(question.document_id, [])
        gold_chunk_ids = map_evidence_to_chunks(question.evidence, document_chunks, threshold)
        for strategy in strategies:
            result = pipeline.ask(
                question.question,
                strategy=strategy,
                document_ids=[question.document_id],
                log_query=False,
            )
            selected_ids = {candidate.chunk.id for candidate in result.selected}
            if gold_chunk_ids:
                evidence_recall = len(gold_chunk_ids & selected_ids) / len(gold_chunk_ids)
                context_precision = len(gold_chunk_ids & selected_ids) / max(len(selected_ids), 1)
            else:
                evidence_recall = float("nan")
                context_precision = float("nan")
            rows.append(
                {
                    "question_id": question.id,
                    "document_id": question.document_id,
                    "split": question.split,
                    "difficulty": question.difficulty or "derived",
                    "strategy": strategy,
                    "answer": result.answer.text,
                    "exact_match": best_reference_score(
                        result.answer.text, question.answers, exact_match
                    ),
                    "token_f1": best_reference_score(
                        result.answer.text, question.answers, token_f1
                    ),
                    "evidence_recall": evidence_recall,
                    "context_precision": context_precision,
                    "gold_evidence_chunks": len(gold_chunk_ids),
                    "selected_chunks": len(result.selected),
                    "selected_tokens": result.selected_tokens,
                    "full_document_tokens": result.full_document_tokens,
                    "token_reduction": result.token_reduction,
                    "retrieval_ms": result.retrieval_ms,
                    "answer_latency_ms": result.answer.latency_ms,
                    "total_ms": result.total_ms,
                    "predicted_budget_label": result.budget.label,
                    "predicted_budget": result.budget.budget,
                    "budget_source": result.budget.source,
                    "budget_expanded": result.budget.expanded,
                    "embedding_backend": result.embedding_backend,
                    "answer_provider": result.answer.provider,
                }
            )

    details = pd.DataFrame(rows)
    grouped = details.groupby("strategy", sort=False)
    summary = grouped.agg(
        questions=("question_id", "count"),
        exact_match=("exact_match", "mean"),
        token_f1=("token_f1", "mean"),
        evidence_recall=("evidence_recall", "mean"),
        context_precision=("context_precision", "mean"),
        median_selected_chunks=("selected_chunks", "median"),
        median_selected_tokens=("selected_tokens", "median"),
        mean_token_reduction=("token_reduction", "mean"),
        mean_total_ms=("total_ms", "mean"),
    ).reset_index()

    bootstrap_samples = int(config["evaluation"].get("bootstrap_samples", 500))
    seed = int(config["project"].get("random_seed", 42))
    intervals = []
    for strategy, group in grouped:
        f1_low, f1_high = _bootstrap_interval(
            group["token_f1"].to_numpy(float), bootstrap_samples, seed
        )
        saving_low, saving_high = _bootstrap_interval(
            group["token_reduction"].to_numpy(float), bootstrap_samples, seed + 1
        )
        intervals.append(
            {
                "strategy": strategy,
                "token_f1_ci_low": f1_low,
                "token_f1_ci_high": f1_high,
                "token_reduction_ci_low": saving_low,
                "token_reduction_ci_high": saving_high,
            }
        )
    summary = summary.merge(pd.DataFrame(intervals), on="strategy", how="left")

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        details.to_csv(destination / "evaluation_details.csv", index=False)
        summary.to_csv(destination / "evaluation_summary.csv", index=False)
        (destination / "evaluation_summary.json").write_text(
            json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8"
        )
        figure = px.scatter(
            summary,
            x="median_selected_tokens",
            y="token_f1",
            color="strategy",
            size="evidence_recall",
            hover_data=["mean_token_reduction", "mean_total_ms"],
            title="Answer quality versus context-token cost",
        )
        figure.write_html(destination / "quality_cost_pareto.html", include_plotlyjs="cdn")
    return details, summary

