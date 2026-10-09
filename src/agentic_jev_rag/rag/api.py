"""The RAG API: Search, Match and Ask as HTTP endpoints that agents call as tools."""

from dataclasses import asdict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from agentic_jev_rag.config import load_config
from agentic_jev_rag.corpus.documents import load_corpus
from agentic_jev_rag.openrouter import OpenRouterError, OpenRouterGateway
from agentic_jev_rag.rag.models import (
    AskRequest,
    AskResponse,
    DocumentOut,
    MatchRequest,
    MatchResponse,
    SearchRequest,
    SearchResponse,
)
from agentic_jev_rag.rag.service import RagService, UnknownDocument
from agentic_jev_rag.system1 import System1Error


def create_rag_app(service: RagService) -> FastAPI:
    app = FastAPI(title="SwiftAI RAG", description="Search, Match and Ask over SwiftAI Recruit's documents.")

    @app.exception_handler(UnknownDocument)
    def unknown(request: Request, error: UnknownDocument):
        return JSONResponse(status_code=404, content={"detail": f"unknown document {error}"})

    @app.exception_handler(System1Error)
    @app.exception_handler(OpenRouterError)
    def upstream(request: Request, error: Exception):
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/search")
    def search(request: SearchRequest) -> SearchResponse:
        return service.search(request)

    @app.post("/match")
    def match(request: MatchRequest) -> MatchResponse:
        return service.match(request)

    @app.post("/ask")
    def ask(request: AskRequest) -> AskResponse:
        return service.ask(request)

    @app.get("/documents/{document_id}")
    def document(document_id: str) -> DocumentOut:
        return DocumentOut(**asdict(service.document(document_id)))

    return app


def create_default_rag_app() -> FastAPI:
    """The app as deployed: config from file and environment, real OpenRouter, the committed corpus."""
    config = load_config()
    service = RagService(
        config, OpenRouterGateway.from_env(), QdrantClient(url=config.qdrant_url), load_corpus(config.corpus_dir)
    )
    return create_rag_app(service)
