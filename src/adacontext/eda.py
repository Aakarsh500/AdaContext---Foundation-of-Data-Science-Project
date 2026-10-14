from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px

from .features import extract_query_features
from .pipeline import AdaptiveContextPipeline
from .schemas import Chunk, QuestionRecord
from .training import map_evidence_to_chunks


def run_eda(
    pipeline: AdaptiveContextPipeline,
    questions: list[QuestionRecord],
    output_dir: str | Path,
) -> dict:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    chunk_rows = [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "document_title": chunk.document_title,
            "section": chunk.section,
            "token_count": chunk.token_count,
        }
        for chunk in pipeline.chunks
    ]
    chunk_frame = pd.DataFrame(chunk_rows)
    document_frame = (
        chunk_frame.groupby(["document_id", "document_title"], as_index=False)
        .agg(chunks=("chunk_id", "count"), total_tokens=("token_count", "sum"))
    )
    chunks_by_document: dict[str, list[Chunk]] = {}
    for chunk in pipeline.chunks:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

    question_rows: list[dict] = []
    score_rows: list[dict] = []
    threshold = float(pipeline.config["evaluation"]["evidence_match_threshold"])
    for question in questions:
        candidates = pipeline.retrieve_candidates(question.question, [question.document_id])
        features = extract_query_features(question.question, candidates)
        gold = map_evidence_to_chunks(
            question.evidence, chunks_by_document.get(question.document_id, []), threshold
        )
        question_rows.append(
            {
                "question_id": question.id,
                "document_id": question.document_id,
                "difficulty": question.difficulty or "unlabelled",
                "split": question.split,
                "gold_evidence_chunks": len(gold),
                **features,
            }
        )
        for rank, candidate in enumerate(candidates, start=1):
            score_rows.append(
                {
                    "question_id": question.id,
                    "rank": rank,
                    "rerank_score": candidate.final_score,
                    "dense_score": candidate.dense_score,
                    "bm25_score": candidate.bm25_score,
                    "is_gold_evidence": candidate.chunk.id in gold,
                }
            )

    question_frame = pd.DataFrame(question_rows)
    score_frame = pd.DataFrame(score_rows)
    chunk_frame.to_csv(destination / "chunks.csv", index=False)
    document_frame.to_csv(destination / "documents.csv", index=False)
    question_frame.to_csv(destination / "questions.csv", index=False)
    score_frame.to_csv(destination / "retrieval_scores.csv", index=False)

    px.bar(
        document_frame,
        x="document_title",
        y="chunks",
        title="Chunks produced per document",
    ).write_html(destination / "chunks_per_document.html", include_plotlyjs="cdn")
    px.histogram(
        chunk_frame,
        x="token_count",
        nbins=30,
        title="Chunk token-count distribution",
    ).write_html(destination / "chunk_token_distribution.html", include_plotlyjs="cdn")
    px.box(
        score_frame,
        x="is_gold_evidence",
        y="rerank_score",
        points="all",
        title="Reranker scores for gold and non-gold chunks",
    ).write_html(destination / "gold_vs_nongold_scores.html", include_plotlyjs="cdn")
    px.scatter(
        question_frame,
        x="score_entropy",
        y="gold_evidence_chunks",
        color="difficulty",
        hover_data=["question_id", "top_score", "score_gap"],
        title="Retrieval uncertainty versus required evidence",
    ).write_html(destination / "uncertainty_vs_evidence.html", include_plotlyjs="cdn")

    summary = {
        "documents": int(document_frame.shape[0]),
        "chunks": int(chunk_frame.shape[0]),
        "questions": int(question_frame.shape[0]),
        "mean_chunks_per_document": float(document_frame["chunks"].mean()),
        "median_chunk_tokens": float(chunk_frame["token_count"].median()),
        "mean_gold_evidence_chunks": float(question_frame["gold_evidence_chunks"].mean()),
        "mean_score_entropy": float(question_frame["score_entropy"].mean()),
        "embedding_backend": pipeline.embedding_backend,
    }
    (destination / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

