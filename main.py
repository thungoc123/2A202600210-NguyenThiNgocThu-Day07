from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def _build_embedder(
    embedding: str,
    local_model: str,
    openai_model: str,
) -> tuple[object, str]:
    selected = (embedding or "auto").strip().lower()
    if selected == "auto":
        selected = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()

    if selected == "local":
        try:
            embedder = LocalEmbedder(model_name=local_model)
            return embedder, ""
        except Exception as exc:
            return _mock_embed, f"[WARN] Local embedder unavailable ({exc}). Falling back to mock."

    if selected == "openai":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            return _mock_embed, "[WARN] OpenAI embedder unavailable (missing OPENAI_API_KEY). Falling back to mock."
        try:
            embedder = OpenAIEmbedder(model_name=openai_model, api_key=api_key)
            return embedder, ""
        except Exception as exc:
            return _mock_embed, f"[WARN] OpenAI embedder unavailable ({exc}). Falling back to mock."

    return _mock_embed, ""


def run_manual_demo(
    question: str | None = None,
    sample_files: list[str] | None = None,
    embedding: str = "auto",
    local_model: str = LOCAL_EMBEDDING_MODEL,
    openai_model: str = OPENAI_EMBEDDING_MODEL,
) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    resolved_local_model = os.getenv("LOCAL_EMBEDDING_MODEL", local_model)
    resolved_openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", openai_model)
    embedder, warning = _build_embedder(
        embedding=embedding,
        local_model=resolved_local_model,
        openai_model=resolved_openai_model,
    )
    if warning:
        print(warning)

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual RAG demo runner.")
    parser.add_argument(
        "question",
        nargs="*",
        help="Optional question string. If omitted, a default prompt is used.",
    )
    parser.add_argument(
        "--embedding",
        choices=["auto", "local", "openai", "mock"],
        default="auto",
        help=(
            "Embedding backend. 'auto' reads EMBEDDING_PROVIDER from env. "
            "Use 'local' for all-MiniLM-L6-v2 and 'openai' for OpenAI embeddings."
        ),
    )
    parser.add_argument(
        "--local-model",
        default=LOCAL_EMBEDDING_MODEL,
        help="Local SentenceTransformer model name.",
    )
    parser.add_argument(
        "--openai-model",
        default=OPENAI_EMBEDDING_MODEL,
        help="OpenAI embedding model name.",
    )

    args = parser.parse_args()
    question = " ".join(args.question).strip() if args.question else None

    return run_manual_demo(
        question=question,
        embedding=args.embedding,
        local_model=args.local_model,
        openai_model=args.openai_model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
