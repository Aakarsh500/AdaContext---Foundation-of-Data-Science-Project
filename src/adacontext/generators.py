from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import AnswerResponse, Candidate
from .text import approximate_token_count, lexical_coverage, split_sentences, token_f1


SYSTEM_PROMPT = """You answer questions using only the supplied document evidence.
Rules:
1. Do not use outside knowledge.
2. If the evidence is insufficient, say: Insufficient evidence in the supplied documents.
3. Be concise but include every figure needed to answer the question.
4. Add inline citations using the exact chunk labels, for example [chunk-id].
5. Do not claim that correlation proves causation."""


def build_prompt(question: str, selected: list[Candidate]) -> str:
    context = "\n\n".join(
        f"[{candidate.chunk.id}]\n"
        f"Document: {candidate.chunk.document_title}\n"
        f"Section: {candidate.chunk.section}\n"
        f"Evidence: {candidate.chunk.text}"
        for candidate in selected
    )
    return f"DOCUMENT EVIDENCE\n{context}\n\nQUESTION\n{question}\n\nANSWER"


class AnswerGenerator(ABC):
    provider: str
    model: str

    @abstractmethod
    def generate(self, question: str, selected: list[Candidate]) -> AnswerResponse:
        raise NotImplementedError


class ExtractiveGenerator(AnswerGenerator):
    """Zero-setup answer baseline that extracts the most question-relevant sentences."""

    def __init__(self, max_sentences: int = 4) -> None:
        self.provider = "extractive"
        self.model = "tfidf-sentence-extractor"
        self.max_sentences = max_sentences

    def generate(self, question: str, selected: list[Candidate]) -> AnswerResponse:
        started = time.perf_counter()
        if not selected:
            return AnswerResponse(
                text="Insufficient evidence in the supplied documents.",
                provider=self.provider,
                model=self.model,
                citations=[],
                input_tokens=approximate_token_count(question),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        records: list[tuple[str, Candidate]] = []
        for candidate in selected:
            for sentence in split_sentences(candidate.chunk.text):
                if len(sentence.split()) >= 4:
                    records.append((sentence, candidate))
        if not records:
            records = [(candidate.chunk.text, candidate) for candidate in selected]

        sentences = [record[0] for record in records]
        try:
            matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(
                [question, *sentences]
            )
            semantic_scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        except ValueError:
            semantic_scores = np.zeros(len(sentences), dtype=float)
        coverage_scores = np.asarray([lexical_coverage(question, sentence) for sentence in sentences])
        numeric_bonus = np.asarray(
            [0.08 if any(character.isdigit() for character in sentence) else 0.0 for sentence in sentences]
        )
        scores = 0.72 * semantic_scores + 0.28 * coverage_scores + numeric_bonus
        order = np.argsort(-scores, kind="stable")

        lower = question.lower()
        broad = any(term in lower for term in ("why", "explain", "compare", "summar", "limitations"))
        desired = self.max_sentences if broad else min(2, self.max_sentences)
        chosen: list[tuple[str, Candidate]] = []
        for index in order:
            sentence, candidate = records[int(index)]
            if chosen and any(token_f1(sentence, existing[0]) >= 0.82 for existing in chosen):
                continue
            chosen.append((sentence, candidate))
            if len(chosen) >= desired:
                break

        answer = " ".join(sentence for sentence, _ in chosen).strip()
        citations = list(dict.fromkeys(candidate.chunk.citation for _, candidate in chosen))
        input_tokens = approximate_token_count(build_prompt(question, selected))
        return AnswerResponse(
            text=answer or "Insufficient evidence in the supplied documents.",
            provider=self.provider,
            model=self.model,
            citations=citations,
            input_tokens=input_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class OllamaGenerator(AnswerGenerator):
    def __init__(self, base_url: str, model: str, temperature: float = 0.0) -> None:
        self.provider = "ollama"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(self, question: str, selected: list[Candidate]) -> AnswerResponse:
        started = time.perf_counter()
        prompt = build_prompt(question, selected)
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": self.temperature},
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("message", {}).get("content", "").strip()
        citations = [candidate.chunk.citation for candidate in selected]
        return AnswerResponse(
            text=text or "Insufficient evidence in the supplied documents.",
            provider=self.provider,
            model=self.model,
            citations=citations,
            input_tokens=int(payload.get("prompt_eval_count") or approximate_token_count(prompt)),
            latency_ms=(time.perf_counter() - started) * 1000,
            raw={"done_reason": payload.get("done_reason")},
        )


class OpenAICompatibleGenerator(AnswerGenerator):
    """Works with LM Studio, vLLM, Ollama's compatible API, or a hosted compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.0,
    ) -> None:
        self.provider = "openai-compatible"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    def generate(self, question: str, selected: list[Candidate]) -> AnswerResponse:
        started = time.perf_counter()
        prompt = build_prompt(question, selected)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"].strip()
        usage = payload.get("usage", {})
        return AnswerResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            citations=[candidate.chunk.citation for candidate in selected],
            input_tokens=int(usage.get("prompt_tokens") or approximate_token_count(prompt)),
            latency_ms=(time.perf_counter() - started) * 1000,
            raw={"finish_reason": payload["choices"][0].get("finish_reason")},
        )


def create_generator(config: dict, provider: str | None = None) -> AnswerGenerator:
    generation = config["generation"]
    chosen = (provider or generation.get("provider", "extractive")).lower()
    if chosen == "extractive":
        return ExtractiveGenerator(int(generation.get("max_answer_sentences", 4)))
    if chosen == "ollama":
        return OllamaGenerator(
            os.getenv("OLLAMA_BASE_URL", generation["ollama_base_url"]),
            os.getenv("OLLAMA_MODEL", generation["ollama_model"]),
            float(generation.get("temperature", 0.0)),
        )
    if chosen in {"compatible", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleGenerator(
            os.getenv("LLM_BASE_URL", generation["compatible_base_url"]),
            os.getenv("LLM_MODEL", generation["compatible_model"]),
            os.getenv("LLM_API_KEY", ""),
            float(generation.get("temperature", 0.0)),
        )
    raise ValueError(f"Unknown generation provider: {chosen}")

