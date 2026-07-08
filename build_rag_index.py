import argparse
import shutil
from pathlib import Path

import numpy as np

from rag.embedding import SentenceTransformerEmbedder
from rag.retrieval import _load_faiss, load_chunks_jsonl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from an already-chunked JSONL corpus."
    )
    parser.add_argument("--chunks-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-base-en-v1.5",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sentence-transformers device override, e.g. cpu or cuda.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.chunks_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks_jsonl(input_path)
    texts = [chunk["text"] for chunk in chunks]

    embedder = SentenceTransformerEmbedder(
        args.embedding_model,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    embeddings = embedder.embed_texts(texts)

    faiss = _load_faiss()
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype=np.float32))

    copied_chunks_path = output_dir / "chunks.jsonl"
    if input_path.resolve() != copied_chunks_path.resolve():
        shutil.copyfile(input_path, copied_chunks_path)

    np.save(output_dir / "embeddings.npy", embeddings)
    faiss.write_index(index, str(output_dir / "faiss.index"))

    print(f"Indexed {len(chunks)} chunks into {output_dir}")


if __name__ == "__main__":
    main()
