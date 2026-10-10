"""Index the committed corpus into Qdrant with the configured embedding model.

Run: uv run python -m agentic_jev_rag.rag.ingest
"""

from qdrant_client import QdrantClient

from agentic_jev_rag.config import load_config
from agentic_jev_rag.corpus.documents import load_corpus
from agentic_jev_rag.openrouter import OpenRouterGateway
from agentic_jev_rag.rag.index import index_corpus


def main() -> None:
    config = load_config()
    documents = load_corpus(config.corpus_dir)
    qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key or None)
    counts = index_corpus(documents, config, OpenRouterGateway.from_env(), qdrant)
    print(f"Indexed {len(documents)} documents into {config.qdrant_url}: {counts}")


if __name__ == "__main__":
    main()
