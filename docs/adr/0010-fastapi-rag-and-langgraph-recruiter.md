# A FastAPI RAG service called as a tool by a LangGraph Recruiter

The RAG is its own FastAPI service exposing Search, Match and Ask. The Recruiter is a LangGraph multi-agent graph behind a second FastAPI service, and its agents reach the RAG only through HTTP tools. This is the stack the target role names, and it keeps the RAG reusable by any agent. Agents call models through our Model gateway rather than LangChain chat models, so per-call cost and latency stay in one code path (ADR 0005) and the gateway stays the single fake in tests. LangGraph supplies the graph, the state, `interrupt` and checkpointing. This replaces ADR 0002's notebooks-first plan: the services are now the package's consumers.

## Considered Options

- An MCP server instead of HTTP: deferred. An MCP adapter over the same endpoints is cheap to add later.
- Python tools inside the agent only: rejected, because it ties the RAG to one agent and doesn't show a service boundary.
