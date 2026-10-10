"""Async HTTP client for the Recruiter API's interview endpoints: the voice worker's only way to the graph."""

import httpx


class RecruiterError(Exception):
    """The Recruiter API failed or could not be reached. `retryable` when resending cannot record an answer twice."""

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


class InterviewRefused(Exception):
    """The interview cannot start: unknown Candidate or Job (404), or no valid Consent (409)."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


class RecruiterClient:
    def __init__(
        self, base_url: str, *, api_key: str = "", http: httpx.AsyncClient | None = None, timeout_s: float = 60.0
    ):
        """`http` lets tests pass an in-process client (httpx.ASGITransport over the Recruiter app)."""
        self._http = http or httpx.AsyncClient(base_url=base_url, timeout=timeout_s)
        self._headers = {"X-API-Key": api_key} if api_key else {}

    async def start(self, candidate_id: str, job_id: str) -> dict:
        response = await self._send("/interviews", {"candidate_id": candidate_id, "job_id": job_id})
        if response.status_code in (404, 409):
            raise InterviewRefused(response.status_code, response.json()["detail"])
        return self._check(response)

    async def reply(self, interview_id: str, message: str, timing: dict | None = None) -> dict:
        body = {"message": message} | ({"timing": timing} if timing else {})
        return self._check(await self._send(f"/interviews/{interview_id}/reply", body))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _send(self, path: str, body: dict) -> httpx.Response:
        try:
            return await self._http.post(path, json=body, headers=self._headers)
        except httpx.ConnectError as error:  # never reached the API: safe to resend
            raise RecruiterError(f"Recruiter API unreachable: {error}", retryable=True) from error
        except httpx.TransportError as error:  # e.g. a timeout: the API may still be processing the answer
            raise RecruiterError(f"Recruiter API did not answer: {error!r}", retryable=False) from error

    @staticmethod
    def _check(response: httpx.Response) -> dict:
        if response.status_code >= 400:
            raise RecruiterError(
                f"Recruiter API {response.request.url.path} → {response.status_code}: {response.text[:300]}",
                retryable=response.status_code >= 500,
            )
        return response.json()
