import pytest


@pytest.mark.parametrize(
    "strategy",
    ["full_context", "bm25_fixed", "dense_fixed", "hybrid_fixed", "elbow", "adaptive"],
)
def test_all_strategies_return_answers(pipeline, strategy):
    result = pipeline.ask(
        "What minimum battery charge is kept during an outage warning?",
        strategy=strategy,
        document_ids=["lakeside-microgrid"],
        fixed_k=3,
        log_query=False,
    )
    assert result.answer.text
    assert result.selected
    assert result.full_document_tokens >= result.selected_tokens
    assert 0.0 <= result.token_reduction <= 1.0


def test_adaptive_context_is_smaller_than_full_context(pipeline):
    question = "When did the Northbridge recycled-water pilot begin?"
    adaptive = pipeline.ask(
        question, strategy="adaptive", document_ids=["northbridge-water"], log_query=False
    )
    full = pipeline.ask(
        question, strategy="full_context", document_ids=["northbridge-water"], log_query=False
    )
    assert adaptive.selected_tokens < full.selected_tokens
    assert adaptive.token_reduction > 0.5
    assert "July 2022" in adaptive.answer.text


def test_empty_question_is_rejected(pipeline):
    with pytest.raises(ValueError, match="empty"):
        pipeline.ask("   ")

