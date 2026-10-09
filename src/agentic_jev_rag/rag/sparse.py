"""Sparse vectors: hashed term frequencies that Qdrant's IDF modifier turns into BM25-style scores (ADR 0011)."""

import hashlib
import re
from collections import Counter

from qdrant_client import models

BUCKETS = 2**20
STOPWORDS = frozenset(
    "a an and are as at be by for from has have i in is it its of on or that the this to was were will with "
    "we you your our their they he she his her my me who what which when where how do does did not no".split()
)


def sparse_vector(text: str) -> models.SparseVector:
    counts: Counter[int] = Counter()
    for token in re.findall(r"\w+", text.lower()):
        if token not in STOPWORDS:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            counts[int.from_bytes(digest) % BUCKETS] += 1
    indices = sorted(counts)
    return models.SparseVector(indices=indices, values=[float(counts[i]) for i in indices])
