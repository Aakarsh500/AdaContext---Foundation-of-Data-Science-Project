from __future__ import annotations

import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

from .schemas import Document, Section
from .text import clean_text, slugify


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "li", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)

    @property
    def text(self) -> str:
        return clean_text("".join(self.parts))


def _sections_from_markdown(text: str) -> list[Section]:
    text = clean_text(text)
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$", flags=re.MULTILINE)
    matches = list(heading_re.finditer(text))
    if not matches:
        return [Section(name="Document", text=text)]

    sections: list[Section] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(name="Preamble", text=preamble))
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append(Section(name=match.group(2).strip(), text=body))
    return sections or [Section(name="Document", text=text)]


def _sections_from_plain_text(text: str) -> list[Section]:
    cleaned = clean_text(text)
    blocks = [block.strip() for block in re.split(r"\n\s*\n", cleaned) if block.strip()]
    if len(blocks) <= 1:
        return [Section(name="Document", text=cleaned)]

    sections: list[Section] = []
    current_name = "Document"
    current_blocks: list[str] = []
    for block in blocks:
        words = block.split()
        looks_like_heading = len(words) <= 12 and not block.endswith((".", "?", "!"))
        if looks_like_heading:
            if current_blocks:
                sections.append(Section(name=current_name, text="\n\n".join(current_blocks)))
                current_blocks = []
            current_name = block
        else:
            current_blocks.append(block)
    if current_blocks:
        sections.append(Section(name=current_name, text="\n\n".join(current_blocks)))
    return sections or [Section(name="Document", text=cleaned)]


def document_from_bytes(data: bytes, filename: str) -> Document:
    suffix = Path(filename).suffix.lower()
    title = Path(filename).stem.replace("_", " ").replace("-", " ").strip() or "Uploaded document"
    document_id = slugify(Path(filename).stem)

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        sections = []
        for page_index, page in enumerate(reader.pages, start=1):
            page_text = clean_text(page.extract_text() or "")
            if page_text:
                sections.append(Section(name=f"Page {page_index}", text=page_text, page=page_index))
        if not sections:
            raise ValueError(f"No extractable text found in PDF: {filename}")
    else:
        text = data.decode("utf-8", errors="replace")
        if suffix in {".html", ".htm"}:
            parser = _ReadableHTMLParser()
            parser.feed(text)
            sections = _sections_from_plain_text(parser.text)
        elif suffix in {".md", ".markdown"}:
            sections = _sections_from_markdown(text)
        elif suffix in {".txt", ""}:
            sections = _sections_from_plain_text(text)
        else:
            raise ValueError(f"Unsupported file type '{suffix}'. Use PDF, TXT, MD, HTML, or JSON.")

    return Document(
        id=document_id,
        title=title,
        sections=sections,
        metadata={"source_filename": filename, "file_type": suffix or ".txt"},
    )


def load_document(path: str | Path) -> Document:
    source = Path(path)
    return document_from_bytes(source.read_bytes(), source.name)


def load_document_stream(stream: BinaryIO, filename: str) -> Document:
    return document_from_bytes(stream.read(), filename)


def load_document_directory(path: str | Path) -> list[Document]:
    directory = Path(path)
    supported = {".pdf", ".txt", ".md", ".markdown", ".html", ".htm"}
    return [load_document(item) for item in sorted(directory.iterdir()) if item.suffix.lower() in supported]


def load_documents_json(path: str | Path) -> list[Document]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    documents: list[Document] = []
    for record in records:
        sections = [
            Section(
                name=section.get("name", "Document"),
                text=clean_text(section.get("text", "")),
                page=section.get("page"),
                metadata=section.get("metadata", {}),
            )
            for section in record.get("sections", [])
            if clean_text(section.get("text", ""))
        ]
        documents.append(
            Document(
                id=str(record["id"]),
                title=str(record.get("title") or record["id"]),
                sections=sections,
                metadata=record.get("metadata", {}),
            )
        )
    return documents

