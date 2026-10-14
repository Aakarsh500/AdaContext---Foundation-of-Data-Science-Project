def test_bundled_dataset_is_complete(sample_data):
    documents, questions = sample_data
    assert len(documents) == 4
    assert len(questions) == 24
    assert {question.split for question in questions} == {"train", "validation", "test"}
    assert all(question.answers for question in questions)
    assert all(question.evidence for question in questions)
    document_ids = {document.id for document in documents}
    assert all(question.document_id in document_ids for question in questions)


def test_bundled_dataset_has_all_difficulty_classes(sample_data):
    _, questions = sample_data
    assert {question.difficulty for question in questions} == {"simple", "moderate", "complex"}

