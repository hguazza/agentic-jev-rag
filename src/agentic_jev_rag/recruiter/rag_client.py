"""HTTP client for the RAG API; the only way agents reach the RAG (ADR 0010)."""

import httpx


class RagError(Exception):
    pass


class RagClient:
    def __init__(self, base_url: str, *, http: httpx.Client | None = None, timeout_s: float = 120.0):
        """`http` lets tests pass an in-process client (FastAPI's TestClient is an httpx.Client)."""
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout_s)

    def search(self, query: str, collections: list[str] | None = None, filters: dict | None = None, top_k: int = 8) -> dict:
        return self._post("/search", {"query": query, "collections": collections, "filters": filters or {}, "top_k": top_k})

    def match(self, job_id: str, top_k: int = 5) -> dict:
        return self._post("/match", {"job_id": job_id, "top_k": top_k})

    def ask(self, question: str, collections: list[str] | None = None, filters: dict | None = None, system_1: str | None = None) -> dict:
        return self._post("/ask", {"question": question, "collections": collections, "filters": filters or {}, "system_1": system_1})

    def document(self, document_id: str) -> dict:
        return self._check(self._http.get(f"/documents/{document_id}"))

    def _post(self, path: str, body: dict) -> dict:
        return self._check(self._http.post(path, json=body))

    @staticmethod
    def _check(response: httpx.Response) -> dict:
        if response.status_code >= 400:
            raise RagError(f"RAG API {response.request.method} {response.request.url.path} → {response.status_code}: {response.text[:300]}")
        return response.json()
