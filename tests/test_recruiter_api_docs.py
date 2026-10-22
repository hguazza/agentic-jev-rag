"""The Recruiter API's OpenAPI document: every operation and every schema field is described."""

import pytest
from fastapi.testclient import TestClient

from agentic_jev_rag.recruiter.api import create_recruiter_app
from agentic_jev_rag.recruiter.schemas import ScreeningVerdict, json_schema

GENERATED = {"HTTPValidationError", "ValidationError"}


@pytest.fixture(scope="module")
def openapi(config):
    from agentic_jev_rag.recruiter.context import RecruiterContext
    from agentic_jev_rag.recruiter.rag_client import RagClient
    from tests.rag_scripts import scripted_gateway

    ctx = RecruiterContext(config=config, gateway=scripted_gateway(), rag=RagClient("http://rag"))
    return TestClient(create_recruiter_app(ctx)).get("/openapi.json").json()


def operations(openapi):
    return {op["operationId"]: op for methods in openapi["paths"].values() for op in methods.values()}


def test_the_app_describes_itself_and_its_tags(openapi):
    assert openapi["info"]["version"] and len(openapi["info"]["description"]) > 200
    assert {t["name"] for t in openapi["tags"]} == {"Recruitment", "Interviews", "Operations"}
    assert all(t["description"] for t in openapi["tags"])


def test_every_operation_is_summarised_described_tagged_and_named(openapi):
    ops = operations(openapi)

    assert set(ops) == {"run_recruitment", "run_status", "start_interview", "reply_to_interview", "health"}
    for operation_id, op in ops.items():
        assert op.get("summary") and len(op.get("description", "")) > 40 and op.get("tags"), operation_id


def test_error_responses_are_documented(openapi):
    ops = operations(openapi)

    assert {"202", "404"} <= set(ops["run_recruitment"]["responses"])
    assert {"404"} <= set(ops["run_status"]["responses"])
    assert {"404", "409", "502"} <= set(ops["start_interview"]["responses"])
    assert {"404", "409", "502"} <= set(ops["reply_to_interview"]["responses"])
    schema = ops["reply_to_interview"]["responses"]["409"]["content"]["application/json"]["schema"]
    assert schema == {"$ref": "#/components/schemas/ErrorOut"}


def test_every_schema_and_field_is_described(openapi):
    for name, schema in openapi["components"]["schemas"].items():
        if name in GENERATED:
            continue
        assert schema.get("description"), name
        for field, spec in schema.get("properties", {}).items():
            assert spec.get("description"), f"{name}.{field}"


def test_request_bodies_offer_named_examples(openapi):
    for path in ("/runs", "/interviews", "/interviews/{interview_id}/reply"):
        content = openapi["paths"][path]["post"]["requestBody"]["content"]["application/json"]
        assert len(content.get("examples", {})) >= 2, path


def test_the_interview_id_parameter_is_described(openapi):
    [parameter] = openapi["paths"]["/interviews/{interview_id}/reply"]["post"]["parameters"]
    assert parameter["description"] and parameter["examples"]


def test_agent_tool_schemas_keep_field_descriptions_for_the_model():
    schema = json_schema(ScreeningVerdict, exclude=("candidate_id", "job_id"))

    assert schema["properties"]["decision"]["description"]
    assert schema["properties"]["scores"]["items"]["properties"]["evidence_chunk_ids"]["description"]


def test_the_api_key_is_a_documented_security_scheme(openapi):
    [scheme] = openapi["components"]["securitySchemes"].values()
    assert scheme == {**scheme, "type": "apiKey", "in": "header", "name": "X-API-Key"}
    assert scheme.get("description")
    ops = operations(openapi)
    assert ops["start_interview"].get("security") and "security" not in ops["health"]
