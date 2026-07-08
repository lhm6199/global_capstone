from __future__ import annotations

from dataclasses import dataclass

from .prompting import RagPromptBuilder
from .retrieval import RagRetriever


@dataclass
class RagChatTurn:
    retriever: RagRetriever
    prompt_builder: RagPromptBuilder
    top_k: int = 3

    def prepare(self, query: str) -> dict:
        search = self.retriever.search_with_stats(query, self.top_k)
        results = search["results"]
        prompt, documents = self.prompt_builder.build_with_documents(query, results)
        return {
            "prompt": prompt,
            "results": results,
            "documents": documents,
            "metrics": search["metrics"],
        }
