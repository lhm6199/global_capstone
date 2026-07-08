from __future__ import annotations

import asyncio
import csv
import io
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .embedding import SentenceTransformerEmbedder

try:
    from minirag import MiniRAG, QueryParam
    from minirag.minirag import always_get_an_event_loop
    from minirag.utils import EmbeddingFunc, encode_string_by_tiktoken
except ImportError:  # pragma: no cover - depends on optional local package install
    MiniRAG = None
    QueryParam = None
    always_get_an_event_loop = None
    EmbeddingFunc = None
    encode_string_by_tiktoken = None


def _single_chunking(content, *_args, **_kwargs):
    token_count = len(encode_string_by_tiktoken(content)) if encode_string_by_tiktoken else 0
    return [{"tokens": token_count, "content": content.strip(), "chunk_order_index": 0}]


def _parse_csv_section(context: str, section_name: str) -> list[list[str]]:
    if not isinstance(context, str) or not context:
        return []
    marker = f"-----{section_name}-----"
    if marker not in context:
        return []
    tail = context.split(marker, 1)[1]
    if "```csv" not in tail:
        return []
    body = tail.split("```csv", 1)[1].split("```", 1)[0].strip()
    if not body:
        return []
    return list(csv.reader(io.StringIO(body)))


@dataclass
class MiniRAGAdapter:
    working_dir: Path
    embedding_model: str
    llm_complete: callable
    chunk_records: list[dict]
    top_k: int = 3
    local_files_only: bool = False
    embedding_batch_size: int = 32
    rebuild_index: bool = False
    embedding_device: str | None = None

    def __post_init__(self):
        if MiniRAG is None:
            raise ImportError(
                "MiniRAG is not importable. Link or install the repo-local `minirag` package first."
            )
        self.working_dir = Path(self.working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._chunk_by_text = {record["text"]: record for record in self.chunk_records}
        self._chunk_by_id = {record["chunk_id"]: record for record in self.chunk_records}
        self._manifest_path = self.working_dir / "index_manifest.json"
        self._embedder = SentenceTransformerEmbedder(
            self.embedding_model,
            batch_size=self.embedding_batch_size,
            local_files_only=self.local_files_only,
            device=self.embedding_device,
        )
        self._rag = self._build_rag()
        self._ensure_index()

    def _build_rag(self):
        async def embed_texts(texts):
            if self.embedding_model.startswith("hash://"):
                return self._embedder.embed_texts(list(texts))
            return await asyncio.to_thread(self._embedder.embed_texts, list(texts))

        return MiniRAG(
            working_dir=str(self.working_dir),
            chunking_func=_single_chunking,
            chunk_overlap_token_size=0,
            chunk_token_size=65536,
            embedding_func=EmbeddingFunc(
                embedding_dim=self._embedder.embedding_dim,
                max_token_size=8192,
                func=embed_texts,
            ),
            llm_model_func=self.llm_complete,
            llm_model_name="qwen3-awq-runtime",
            llm_model_max_token_size=32768,
        )

    def _expected_manifest(self):
        return {
            "embedding_model": self.embedding_model,
            "chunk_count": len(self.chunk_records),
            "chunk_ids": [record["chunk_id"] for record in self.chunk_records],
        }

    def _ensure_index(self):
        expected = self._expected_manifest()
        if self.rebuild_index and self.working_dir.exists():
            shutil.rmtree(self.working_dir)
            self.working_dir.mkdir(parents=True, exist_ok=True)
            self._rag = self._build_rag()

        if self._manifest_path.exists():
            manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            if manifest == expected:
                return
            raise ValueError(
                f"MiniRAG index at {self.working_dir} does not match the current chunk set. "
                "Use --rebuild_minirag_index to rebuild it."
            )

        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self._rag.ainsert(
                [record["text"] for record in self.chunk_records],
                ids=[record["chunk_id"] for record in self.chunk_records],
            )
        )
        self._manifest_path.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_index_stats(self) -> dict:
        graph_path = self.working_dir / "graph_chunk_entity_relation.graphml"
        entity_vdb_path = self.working_dir / "vdb_entities.json"
        relation_vdb_path = self.working_dir / "vdb_relationships.json"
        chunk_vdb_path = self.working_dir / "vdb_chunks.json"
        return {
            "working_dir": str(self.working_dir),
            "chunk_count": len(self.chunk_records),
            "manifest_path": str(self._manifest_path),
            "graph_path": str(graph_path),
            "graph_exists": graph_path.exists(),
            "entity_vdb_exists": entity_vdb_path.exists(),
            "relationship_vdb_exists": relation_vdb_path.exists(),
            "chunk_vdb_exists": chunk_vdb_path.exists(),
        }

    def _build_retrieved_from_context(self, context: str) -> list[dict]:
        rows = _parse_csv_section(context, "Sources")
        if not rows:
            return []
        header, *values = rows
        retrieved = []
        for row in values:
            if len(row) != len(header):
                continue
            item = dict(zip(header, row))
            content = item.get("content", "").strip()
            if not content:
                continue
            source = self._chunk_by_text.get(content)
            if source is None:
                continue
            retrieved.append(
                {
                    "chunk_id": source["chunk_id"],
                    "title": source.get("title"),
                    "text": source["text"],
                    "source": source.get("source"),
                }
            )
        return retrieved

    def prepare(self, question: str, mode: str) -> dict:
        started = time.perf_counter()
        if mode == "naive":
            loop = always_get_an_event_loop()
            results = loop.run_until_complete(
                self._rag.chunks_vdb.query(question, top_k=self.top_k)
            )
            chunk_ids = [item["id"] for item in results]
            chunks = loop.run_until_complete(self._rag.text_chunks.get_by_ids(chunk_ids))
            retrieved = []
            contexts = []
            for rank, chunk in enumerate(chunks, start=1):
                if chunk is None:
                    continue
                source = self._chunk_by_id.get(chunk["full_doc_id"])
                if source is None:
                    continue
                retrieved.append(
                    {
                        "rank": rank,
                        "score": float(results[rank - 1].get("distance", 0.0)),
                        "chunk_id": source["chunk_id"],
                        "title": source.get("title"),
                        "text": source["text"],
                        "source": source.get("source"),
                    }
                )
                contexts.append(source["text"])
            context = "\n--New Chunk--\n".join(contexts)
        else:
            context = self._rag.query(
                question,
                param=QueryParam(mode=mode, only_need_context=True, top_k=self.top_k),
            )
            retrieved = self._build_retrieved_from_context(context)

        retrieval_ms = (time.perf_counter() - started) * 1000.0
        return {
            "context": context or "",
            "retrieved": retrieved[: self.top_k],
            "metrics": {"minirag_retrieval_ms": retrieval_ms},
        }
