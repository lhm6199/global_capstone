from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedding import SentenceTransformerEmbedder


def _load_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS is required for RAG retrieval. Install a platform-compatible "
            "package such as `faiss-cpu` or a Jetson-specific build."
        ) from exc
    return faiss


def load_chunks_jsonl(path: str | Path) -> list[dict]:
    chunks_path = Path(path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunks_path}")

    chunks = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {chunks_path}: {exc}"
                ) from exc
            text = chunk.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Chunk line {line_number} in {chunks_path} is missing a non-empty `text` field."
                )
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")
    return chunks


@dataclass
class RagRetriever:
    embedder: SentenceTransformerEmbedder
    index: object
    chunks: list[dict]

    @classmethod
    def from_index_dir(
        cls,
        index_dir: str | Path,
        embedding_model: str,
        *,
        batch_size: int = 32,
        local_files_only: bool = False,
        device: str | None = None,
    ) -> "RagRetriever":
        index_path = Path(index_dir)
        chunks = load_chunks_jsonl(index_path / "chunks.jsonl")
        faiss_index_path = index_path / "faiss.index"
        if not faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")

        embedder = SentenceTransformerEmbedder(
            embedding_model,
            batch_size=batch_size,
            local_files_only=local_files_only,
            device=device,
        )
        faiss = _load_faiss()
        index = faiss.read_index(str(faiss_index_path))
        if index.ntotal != len(chunks):
            raise ValueError(
                f"Index/document count mismatch: index has {index.ntotal}, chunks file has {len(chunks)}"
            )

        return cls(embedder=embedder, index=index, chunks=chunks)

    def search(self, query: str, top_k: int) -> list[dict]:
        return self.search_with_stats(query, top_k)["results"]

    def search_with_stats(self, query: str, top_k: int) -> dict:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        embed_started = time.perf_counter()
        query_vector = np.asarray([self.embedder.embed_query(query)], dtype=np.float32)
        embed_query_ms = (time.perf_counter() - embed_started) * 1000.0

        search_started = time.perf_counter()
        scores, indices = self.index.search(query_vector, top_k)
        faiss_search_ms = (time.perf_counter() - search_started) * 1000.0
        results = []
        for rank, (score, chunk_idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if chunk_idx < 0 or chunk_idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[chunk_idx])
            chunk["rank"] = rank
            chunk["score"] = float(score)
            results.append(chunk)
        return {
            "results": results,
            "metrics": {
                "embed_query_ms": embed_query_ms,
                "faiss_search_ms": faiss_search_ms,
            },
        }
