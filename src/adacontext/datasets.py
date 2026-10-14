from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from .ingest import load_documents_json
from .schemas import Document, QuestionRecord, Section
from .text import clean_text


QASPER_PARQUET_API = "https://huggingface.co/api/datasets/allenai/qasper/parquet"


def load_questions_jsonl(path: str | Path) -> list[QuestionRecord]:
    questions: list[QuestionRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            questions.append(
                QuestionRecord(
                    id=str(record["id"]),
                    document_id=str(record["document_id"]),
                    question=str(record["question"]),
                    answers=[str(value) for value in record.get("answers", [])],
                    evidence=[str(value) for value in record.get("evidence", [])],
                    difficulty=record.get("difficulty"),
                    split=str(record.get("split", "test")),
                    metadata=record.get("metadata", {}),
                )
            )
    return questions


def save_questions_jsonl(questions: Iterable[QuestionRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question.to_dict(), ensure_ascii=False) + "\n")


def load_sample_dataset(project_root: str | Path) -> tuple[list[Document], list[QuestionRecord]]:
    base = Path(project_root) / "data" / "sample"
    return load_documents_json(base / "documents.json"), load_questions_jsonl(
        base / "questions.jsonl"
    )


def download_qasper(output_dir: str | Path, force: bool = False) -> list[Path]:
    """Download the official Hugging Face auto-converted QASPER Parquet files."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_response = requests.get(QASPER_PARQUET_API, timeout=60)
    manifest_response.raise_for_status()
    manifest = manifest_response.json()
    configuration = manifest.get("qasper") or next(iter(manifest.values()))
    downloaded: list[Path] = []
    for split, urls in configuration.items():
        if not urls:
            continue
        target = destination / f"{split}.parquet"
        if target.exists() and not force:
            downloaded.append(target)
            continue
        temporary = target.with_suffix(".parquet.part")
        with requests.get(urls[0], stream=True, timeout=180) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
        temporary.replace(target)
        downloaded.append(target)
    return downloaded


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _deduplicate(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(str(value or ""))
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def _normalize_qasper_answer_group(group: dict) -> tuple[list[str], list[str], list[str]]:
    answers: list[str] = []
    evidence: list[str] = []
    answer_types: list[str] = []
    for annotation in _as_list(group.get("answer")):
        if not isinstance(annotation, dict):
            continue
        if bool(annotation.get("unanswerable")):
            answers.append("Unanswerable")
            answer_types.append("unanswerable")
        elif annotation.get("yes_no") is not None:
            answers.append("Yes" if bool(annotation["yes_no"]) else "No")
            answer_types.append("yes_no")
        else:
            spans = _deduplicate(_as_list(annotation.get("extractive_spans")))
            free_form = clean_text(str(annotation.get("free_form_answer") or ""))
            if spans:
                answers.append(" ".join(spans))
                answer_types.append("extractive")
            if free_form:
                answers.append(free_form)
                answer_types.append("abstractive")
        highlighted = _as_list(annotation.get("highlighted_evidence"))
        evidence.extend(highlighted or _as_list(annotation.get("evidence")))
    return _deduplicate(answers), _deduplicate(evidence), _deduplicate(answer_types)


def load_qasper(
    parquet_dir: str | Path,
    splits: Iterable[str] | None = None,
    max_documents_per_split: int | None = None,
) -> tuple[list[Document], list[QuestionRecord]]:
    """Normalize QASPER's document-centric records into AdaContext schemas."""

    directory = Path(parquet_dir)
    requested = list(splits or ("train", "validation", "test"))
    documents: list[Document] = []
    questions: list[QuestionRecord] = []
    for split in requested:
        path = directory / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `adacontext download-qasper` before loading QASPER."
            )
        frame = pd.read_parquet(path)
        if max_documents_per_split is not None:
            frame = frame.head(max_documents_per_split)
        for row in frame.itertuples(index=False):
            paper_id = str(row.id)
            full_text = row.full_text or {}
            section_names = _as_list(full_text.get("section_name"))
            paragraph_groups = _as_list(full_text.get("paragraphs"))
            sections: list[Section] = []
            abstract = clean_text(str(row.abstract or ""))
            if abstract:
                sections.append(Section(name="Abstract", text=abstract))
            for section_index, paragraphs in enumerate(paragraph_groups):
                name = (
                    str(section_names[section_index])
                    if section_index < len(section_names)
                    else f"Section {section_index + 1}"
                )
                body = clean_text("\n\n".join(str(item) for item in _as_list(paragraphs)))
                if body:
                    sections.append(Section(name=name, text=body))
            documents.append(
                Document(
                    id=paper_id,
                    title=clean_text(str(row.title or paper_id)),
                    sections=sections,
                    metadata={"dataset": "QASPER", "split": split},
                )
            )

            qas = row.qas or {}
            q_texts = _as_list(qas.get("question"))
            q_ids = _as_list(qas.get("question_id"))
            q_answers = _as_list(qas.get("answers"))
            for question_index, question_text in enumerate(q_texts):
                group = q_answers[question_index] if question_index < len(q_answers) else {}
                answers, evidence, answer_types = _normalize_qasper_answer_group(group or {})
                questions.append(
                    QuestionRecord(
                        id=str(q_ids[question_index]) if question_index < len(q_ids) else f"{paper_id}-{question_index}",
                        document_id=paper_id,
                        question=clean_text(str(question_text)),
                        answers=answers or ["Unanswerable"],
                        evidence=evidence,
                        difficulty=None,
                        split="validation" if split == "validation" else split,
                        metadata={"dataset": "QASPER", "answer_types": answer_types},
                    )
                )
    return documents, questions

