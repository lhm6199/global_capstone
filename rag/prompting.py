from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROMPT_TEMPLATE = Path(__file__).with_name("templates") / "default_rag_prompt.txt"


def _string_value(value, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


@dataclass
class RagPromptBuilder:
    template_text: str

    @classmethod
    def from_file(cls, template_path: str | Path | None = None) -> "RagPromptBuilder":
        path = Path(template_path) if template_path else DEFAULT_PROMPT_TEMPLATE
        return cls(template_text=path.read_text(encoding="utf-8"))

    def render_documents(self, results: list[dict]) -> str:
        blocks = []
        for item in results:
            score = item.get("score")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            blocks.append(
                "\n".join(
                    [
                        f"[Document {item.get('rank', '?')}]",
                        f"title: {_string_value(item.get('title'), 'untitled')}",
                        f"chunk_id: {_string_value(item.get('chunk_id'), 'n/a')}",
                        f"source: {_string_value(item.get('source'), 'n/a')}",
                        f"score: {score_text}",
                        "text:",
                        item["text"].strip(),
                    ]
                )
            )
        return "\n\n".join(blocks) if blocks else "[No retrieved documents]"

    def build(self, query: str, results: list[dict]) -> str:
        return self.template_text.format(
            documents=self.render_documents(results),
            question=query.strip(),
        )

    def build_with_documents(self, query: str, results: list[dict]) -> tuple[str, str]:
        documents = self.render_documents(results)
        prompt = self.template_text.format(
            documents=documents,
            question=query.strip(),
        )
        return prompt, documents
