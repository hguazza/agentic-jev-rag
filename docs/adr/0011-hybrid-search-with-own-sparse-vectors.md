# Hybrid search with our own sparse vectors and Qdrant's IDF modifier

Sparse vectors are hashed term frequencies computed in our code, and Qdrant's `modifier: idf` turns them into BM25-style scores at query time. Dense and sparse results are fused in Qdrant with RRF. This avoids adding a sparse-embedding model dependency, keeps tests deterministic, and leaves the dense side as the only place embeddings come from (the gateway).

## Considered Options

- FastEmbed's BM25 or SPLADE models: rejected for v1, because they add an ONNX runtime dependency for little gain on a small corpus.
