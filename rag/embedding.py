from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class HashingEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                token_hash = hash(token)
                index = token_hash % self.dim
                sign = 1.0 if ((token_hash >> 1) & 1) == 0 else -1.0
                vectors[row, index] += sign
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors


@dataclass
class SentenceTransformerEmbedder:
    model_name: str
    batch_size: int = 32
    local_files_only: bool = False
    device: str | None = None

    def __post_init__(self):
        if self.model_name.startswith("hash://"):
            try:
                self._model = HashingEmbedder(dim=int(self.model_name.split("://", 1)[1]))
                self.embedding_dim = self._model.dim
            except ValueError as exc:
                raise ValueError(
                    "Hash embedder model names must look like `hash://256`."
                ) from exc
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for RAG embedding. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        model_kwargs = {"local_files_only": self.local_files_only}
        if self.device:
            model_kwargs["device"] = self.device
        self._model = SentenceTransformer(self.model_name, **model_kwargs)
        self.embedding_dim = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if hasattr(self._model, "embed_texts"):
            return np.asarray(self._model.embed_texts(texts), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
