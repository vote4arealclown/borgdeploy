"""Tests for process-pool CPU worker."""
from __future__ import annotations

import time

from borg.cpu_worker import EmbeddingWorker


def test_process_pool_embedding() -> None:
    """Process pool should embed a batch of texts."""
    worker = EmbeddingWorker(num_workers=2, dims=16)
    texts = [f"Market analysis {i}" for i in range(10)]

    start = time.perf_counter()
    embeddings = worker.embed_texts(texts)
    elapsed = time.perf_counter() - start

    assert len(embeddings) == len(texts)
    for emb in embeddings:
        assert len(emb) == 16

    # All embeddings should be distinct (highly likely with deterministic seeding).
    assert len(set(tuple(e) for e in embeddings)) == len(texts)
    print(f"Batch embedding took {elapsed:.3f}s")


def test_process_pool_correctness() -> None:
    """Process pool results must match the serial deterministic implementation."""
    worker = EmbeddingWorker(num_workers=2, dims=64)
    texts = [f"Text number {i} with some content" for i in range(20)]

    from borg.cpu_worker import _embed_batch_worker

    serial = _embed_batch_worker(texts, dims=64)
    parallel = worker.embed_texts(texts)

    assert len(serial) == len(parallel)
    assert serial == parallel
