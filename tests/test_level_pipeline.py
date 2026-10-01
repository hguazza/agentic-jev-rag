"""Level pipeline seam: `answer()` against real Qdrant with a fake Model gateway."""

from agentic_jev_rag.pipeline import answer

FACEBOOK_IRELAND = "62019CJ0645"
SCHREMS = "62014CJ0362"


def test_level_1_answer_cites_chunks_from_the_fixture_document_that_answers_it(
    indexed_case_law, config, gateway
):
    record = answer(
        "Under the one-stop shop mechanism, may a supervisory authority that is not the "
        "lead supervisory authority bring cross-border processing before a national court?",
        level=1,
        config=config,
        gateway=gateway,
    )

    assert record.answer
    assert record.citations
    assert all(c.chunk_id in record.retrieved_chunk_ids for c in record.citations)
    assert record.citations[0].document_id == FACEBOOK_IRELAND


def test_level_1_retrieves_the_document_that_matches_the_question(indexed_case_law, config, gateway):
    record = answer(
        "Was Decision 2000/520 on the adequacy of the safe harbour privacy "
        "principles for transfers to the United States declared invalid?",
        level=1,
        config=config,
        gateway=gateway,
    )

    assert record.citations[0].document_id == SCHREMS


def test_level_1_makes_no_system_1_decisions(indexed_case_law, config, gateway):
    record = answer(
        "What is the one-stop shop mechanism?", level=1, config=config, gateway=gateway
    )

    assert record.system_1_trace == []


def test_level_1_sends_retrieved_chunks_to_the_system_2_model(
    indexed_case_law, config, gateway
):
    answer("What is the one-stop shop mechanism?", level=1, config=config, gateway=gateway)

    [call] = gateway.completion_calls
    assert call.model == config.models.system_2_model
    assert "one-stop shop" in call.prompt.lower()
