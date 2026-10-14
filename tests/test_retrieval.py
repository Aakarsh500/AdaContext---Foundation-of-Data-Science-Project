def test_hybrid_retrieval_finds_economic_evidence(pipeline):
    candidates = pipeline.retrieve_candidates(
        "Why would the Riverside connection improve the payback period?",
        ["northbridge-water"],
    )
    assert candidates
    assert candidates[0].chunk.section == "Economics and expansion"
    assert "9.4 years" in candidates[0].chunk.text


def test_hybrid_retrieval_finds_health_threshold(pipeline):
    candidates = pipeline.retrieve_candidates(
        "What temperature defined a qualifying heat day?",
        ["cedar-heat"],
    )
    top_evidence = " ".join(candidate.chunk.text for candidate in candidates[:2])
    assert "38" in top_evidence and "degree" in top_evidence
    assert candidates[0].dense_rank is not None
    assert candidates[0].bm25_rank is not None


def test_document_filter_is_respected(pipeline):
    candidates = pipeline.retrieve_candidates("What was the payback period?", ["lakeside-microgrid"])
    assert candidates
    assert all(candidate.chunk.document_id == "lakeside-microgrid" for candidate in candidates)
