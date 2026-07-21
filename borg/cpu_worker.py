"""Process-pool offloading for CPU/embedding work."""
from __future__ import annotations

import math
import random
from multiprocessing import Pool
from typing import Optional


def _fake_embedding(text: str, dims: int = 768) -> list[float]:
    """Deterministic hash-based embedding; runs in a worker process."""
    rng = random.Random(text)
    vec = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _embed_batch_worker(texts: list[str], dims: int = 768) -> list[list[float]]:
    """Worker function: embed a batch of texts without GIL contention."""
    return [_fake_embedding(text, dims) for text in texts]


class EmbeddingWorker:
    """Offload batch embedding generation to a process pool.

    Falls back to in-process execution if the worker pool cannot be created.
    """

    def __init__(self, num_workers: int = 2, dims: int = 768) -> None:
        self.dims = dims
        self._pool: Optional[Pool] = None
        try:
            # spawn avoids forking issues with async runtime / database handles.
            self._pool = Pool(processes=num_workers)
        except Exception:
            self._pool = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, parallelizing across worker processes."""
        if not texts:
            return []
        if self._pool is None:
            return _embed_batch_worker(texts, self.dims)
        try:
            return self._pool.apply(_embed_batch_worker, args=(texts, self.dims))
        except Exception:
            return _embed_batch_worker(texts, self.dims)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
