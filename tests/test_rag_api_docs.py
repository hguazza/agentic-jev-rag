"""The RAG API's OpenAPI document: every operation and every schema field is described."""

import pytest
from fastapi.testclient import TestClient

from agentic_jev_rag.rag.api import create_rag_app
from agentic_jev_rag.rag.service import RagService
from tests.rag_scripts import scripted_gateway

GENERATED = {"HTTPValidationError", "ValidationError"}  # FastAPI's own 422 schemas


@pytest.fixture(scope="module")
def openapi(config, qdrant):
    service = RagService(config, scripted_gateway(), qdrant, [])
    return TestClient(create_rag_app(service)).get("/openapi.json").json()


def operations(openapi):
    return [(path, method, op) for path, methods in openapi["paths"].items() for method, op in methods.items()]


def test_the_app_describes_itself_and_its_tags(openapi):
    info = openapi["info"]
    assert info["version"] and len(info["description"]) > 200
    assert {t["name"] for t in openapi["tags"]} == {"Retrieval", "Documents", "Operations"}
    assert all(t["description"] for t in openapi["tags"])


def test_every_operation_has_a_summary_description_tag_and_stable_operation_id(openapi):
    for path, method, op in operations(openapi):
        assert op.get("summary"), (method, path)
        assert len(op.get("description", "")) > 40, (method, path)
        assert op.get("tags"), (method, path)
    ids = {op["operationId"] for _, _, op in operations(openapi)}
    assert ids == {"search", "match_candidates", "ask", "get_document", "health"}


def test_error_responses_are_documented_with_a_typed_body(openapi):
    by_id = {op["operationId"]: op for _, _, op in operations(openapi)}

    for operation_id in ("match_candidates", "get_document", "search", "ask"):
        assert "404" in by_id[operation_id]["responses"], operation_id
    for operation_id in ("search", "match_candidates", "ask"):
        assert "502" in by_id[operation_id]["responses"], operation_id
    error = by_id["get_document"]["responses"]["404"]["content"]["application/json"]["schema"]
    assert error == {"$ref": "#/components/schemas/ErrorOut"}


def test_every_schema_and_field_is_described(openapi):
    for name, schema in openapi["components"]["schemas"].items():
        if name in GENERATED:
            continue
        assert schema.get("description"), name
        for field, spec in schema.get("properties", {}).items():
            assert spec.get("description"), f"{name}.{field}"


def test_request_bodies_offer_named_examples(openapi):
    for path in ("/search", "/match", "/ask"):
        content = openapi["paths"][path]["post"]["requestBody"]["content"]["application/json"]
        assert len(content.get("examples", {})) >= 2, path


def test_the_document_id_path_parameter_is_described(openapi):
    [parameter] = openapi["paths"]["/documents/{document_id}"]["get"]["parameters"]
    assert parameter["description"] and parameter["examples"]
