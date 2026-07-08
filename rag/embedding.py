from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SentenceTransformerEmbedder:
    model_name: str
    batch_size: int = 32
    local_files_only: bool = False
    device: str | None = None

    def __post_init__(self):
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

    def embed_texts(self, texts: list[str]) -> np.ndarray:
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
