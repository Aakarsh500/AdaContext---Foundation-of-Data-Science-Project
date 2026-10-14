from adacontext.chunking import StructureAwareChunker
from adacontext.schemas import Document, Section


def test_chunker_preserves_document_and_section_metadata():
    document = Document(
        id="doc",
        title="A test document",
        sections=[
            Section(
                name="Methods",
                page=3,
                text=" ".join(
                    f"Sentence {index} explains a repeatable experimental method."
                    for index in range(1, 25)
                ),
            )
        ],
    )
    chunks = StructureAwareChunker(
        target_tokens=45,
        overlap_tokens=8,
        min_chunk_tokens=15,
        max_chunk_tokens=70,
    ).chunk_document(document)
    assert len(chunks) > 1
    assert all(chunk.document_id == "doc" for chunk in chunks)
    assert all(chunk.section == "Methods" for chunk in chunks)
    assert all(chunk.page == 3 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_short_section_becomes_one_chunk():
    document = Document(
        id="short",
        title="Short",
        sections=[Section(name="Summary", text="The measured reduction was 21 percent.")],
    )
    chunks = StructureAwareChunker().chunk_document(document)
    assert len(chunks) == 1
    assert "21 percent" in chunks[0].text

