from __future__ import annotations

from itertools import cycle
from pathlib import Path

import pandas as pd
import plotly.express as px

from .pipeline import AdaptiveContextPipeline
from .schemas import Document, QuestionRecord, Section
from .text import token_f1
from .training import evidence_match_score


def run_lost_in_middle_stress_test(
    config: dict,
    source_documents: list[Document],
    questions: list[QuestionRecord],
    output_dir: str | Path,
    context_lengths: tuple[int, ...] = (10, 25, 50),
    maximum_questions: int = 6,
) -> pd.DataFrame:
    """Create controlled distractor documents with evidence at start, middle, or end."""

    usable = [question for question in questions if question.evidence and question.answers]
    usable = usable[:maximum_questions]
    distractors = [
        section.text
        for document in source_documents
        for section in document.sections
        if section.text
    ]
    if not usable or len(distractors) < 2:
        raise ValueError("Stress testing requires questions with evidence and multiple source sections.")

    synthetic_documents: list[Document] = []
    synthetic_questions: list[QuestionRecord] = []
    positions = ("start", "middle", "end")
    for question in usable:
        other = [
            text
            for document in source_documents
            if document.id != question.document_id
            for text in [section.text for section in document.sections]
        ] or distractors
        for length in context_lengths:
            for position in positions:
                insert_at = {"start": 0, "middle": length // 2, "end": length - 1}[position]
                filler = cycle(other)
                sections: list[Section] = []
                for index in range(length):
                    if index == insert_at:
                        text = " ".join(question.evidence)
                        name = "Target evidence"
                    else:
                        text = next(filler)
                        name = f"Distractor {index + 1}"
                    sections.append(Section(name=name, text=text))
                synthetic_id = f"stress-{question.id}-{length}-{position}"
                synthetic_documents.append(
                    Document(
                        id=synthetic_id,
                        title=f"Stress case {question.id}: {length} chunks, {position}",
                        sections=sections,
                        metadata={"context_length": length, "evidence_position": position},
                    )
                )
                synthetic_questions.append(
                    QuestionRecord(
                        id=synthetic_id,
                        document_id=synthetic_id,
                        question=question.question,
                        answers=question.answers,
                        evidence=question.evidence,
                        difficulty=question.difficulty,
                        metadata={
                            "context_length": length,
                            "evidence_position": position,
                            "source_question_id": question.id,
                        },
                    )
                )

    pipeline = AdaptiveContextPipeline(config).index_documents(synthetic_documents)
    rows: list[dict] = []
    for question in synthetic_questions:
        for strategy in ("full_context", "adaptive"):
            result = pipeline.ask(
                question.question,
                strategy=strategy,
                document_ids=[question.document_id],
                log_query=False,
            )
            best_answer_f1 = max(token_f1(result.answer.text, answer) for answer in question.answers)
            evidence_selected = any(
                max(evidence_match_score(span, candidate.chunk.text) for span in question.evidence)
                >= float(config["evaluation"]["evidence_match_threshold"])
                for candidate in result.selected
            )
            rows.append(
                {
                    "source_question_id": question.metadata["source_question_id"],
                    "context_chunks": question.metadata["context_length"],
                    "evidence_position": question.metadata["evidence_position"],
                    "strategy": strategy,
                    "evidence_selected": float(evidence_selected),
                    "answer_token_f1": best_answer_f1,
                    "selected_tokens": result.selected_tokens,
                    "token_reduction": result.token_reduction,
                    "total_ms": result.total_ms,
                }
            )

    frame = pd.DataFrame(rows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "lost_in_middle_details.csv", index=False)
    summary = (
        frame.groupby(["context_chunks", "evidence_position", "strategy"], as_index=False)
        .agg(
            evidence_recall=("evidence_selected", "mean"),
            answer_token_f1=("answer_token_f1", "mean"),
            median_selected_tokens=("selected_tokens", "median"),
        )
    )
    summary.to_csv(destination / "lost_in_middle_summary.csv", index=False)
    figure = px.line(
        summary,
        x="context_chunks",
        y="answer_token_f1",
        color="strategy",
        line_dash="evidence_position",
        markers=True,
        title="Lost-in-the-middle stress test",
    )
    figure.write_html(destination / "lost_in_middle.html", include_plotlyjs="cdn")
    return summary

