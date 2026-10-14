from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
APPROX_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it",
    "its", "of", "on", "or", "that", "the", "their", "then", "there", "these", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "will", "with",
}


def clean_text(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n[ ]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def lexical_tokens(text: str, remove_stopwords: bool = False) -> list[str]:
    tokens = [token.lower().replace("’", "'") for token in TOKEN_RE.findall(text or "")]
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return tokens


def approximate_token_count(text: str) -> int:
    return len(APPROX_TOKEN_RE.findall(text or ""))


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(cleaned) if part.strip()]
    return sentences or [cleaned]


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    common = Counter(predicted) & Counter(gold)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def lexical_coverage(question: str, text: str) -> float:
    query_terms = set(lexical_tokens(question, remove_stopwords=True))
    if not query_terms:
        return 0.0
    text_terms = set(lexical_tokens(text, remove_stopwords=True))
    return len(query_terms & text_terms) / len(query_terms)


def slugify(value: str, fallback: str = "document") -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or fallback

