from adacontext.text import exact_match, token_f1
from adacontext.training import map_evidence_to_chunks


def test_answer_metrics_normalize_articles_and_punctuation():
    assert exact_match("The July 2022.", "July 2022") == 1.0
    assert token_f1("reduction was 21 percent", "21 percent reduction") > 0.7


def test_evidence_mapping_finds_gold_chunk(pipeline):
    chunks = [chunk for chunk in pipeline.chunks if chunk.document_id == "oakridge-library"]
    gold = map_evidence_to_chunks(
        ["The audit used WCAG 2.2 level AA as its target"],
        chunks,
        threshold=0.55,
    )
    assert len(gold) == 1
    matched = next(chunk for chunk in chunks if chunk.id in gold)
    assert matched.section == "Audit scope"

