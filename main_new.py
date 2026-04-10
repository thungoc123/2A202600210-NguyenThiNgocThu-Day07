from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.chunking import SemanticMetadataChunker
from src.embeddings import (
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore


@dataclass
class BenchmarkItem:
    qid: str
    question: str
    gold_answer: str
    expected_sources: list[str]
    requires_filter: bool


def normalize_text(text: str) -> str:
    lowered = text.lower().replace("đ", "d").strip()
    no_accent = (
        unicodedata.normalize("NFKD", lowered)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"[^a-z0-9]+", " ", no_accent).strip()


def canonical_source_key(name: str) -> str:
    norm = normalize_text(name)
    compact = norm.replace(" ", "")

    if "luat bien viet nam" in norm or "luatbienvietnam" in compact:
        return "luatbienvietnam"
    if "144" in norm and "nghidinh" in compact:
        return "nghidinh144"
    if "dieu kien kinh doanh" in norm and "nghidinh" in compact:
        return "nghidinhvedieukienkinhdoanh"
    if "van tai da phuong thuc" in norm and "nghidinh" in compact:
        return "nghidinhvevantai"
    if "thong tu 66" in norm or "66 2014" in norm or "tt bgtvt" in norm:
        return "thongtu66"

    return compact


def expected_match(expected_sources: list[str], actual_source_name: str) -> bool:
    actual_key = canonical_source_key(Path(actual_source_name).stem)
    if not actual_key:
        return False

    for expected in expected_sources:
        expected_key = canonical_source_key(Path(expected).stem)
        if not expected_key:
            continue
        if expected_key in actual_key or actual_key in expected_key:
            return True
    return False


def parse_benchmark_md(md_path: Path) -> list[BenchmarkItem]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    items: list[BenchmarkItem] = []

    for line in lines:
        raw = line.strip()
        if not raw.startswith("| Q"):
            continue

        cols = [c.strip() for c in raw.strip("|").split("|")]
        if len(cols) < 6:
            continue

        qid = cols[0]
        question = cols[1]
        gold_answer = cols[2]
        source_cell = cols[4]
        filter_cell = cols[5]

        source_parts = re.split(r"\s*&\s*|,\s*", source_cell)
        expected_sources = [s.strip().strip("`") for s in source_parts if s.strip()]

        requires_filter = "co" in normalize_text(filter_cell)

        items.append(
            BenchmarkItem(
                qid=qid,
                question=question,
                gold_answer=gold_answer,
                expected_sources=expected_sources,
                requires_filter=requires_filter,
            )
        )

    return items


def infer_doc_type(file_name: str) -> str:
    key = normalize_text(file_name)
    if "nghidinh" in key:
        return "decree"
    if "thongtu" in key:
        return "circular"
    if "luat" in key:
        return "law"
    return "other"


def build_chunker(args: argparse.Namespace):
    if args.strategy == "fixed":
        return FixedSizeChunker(chunk_size=args.chunk_size, overlap=args.overlap)
    if args.strategy == "sentence":
        return SentenceChunker(max_sentences_per_chunk=args.max_sentences)
    if args.strategy == "semantic_metadata":
        return SemanticMetadataChunker(
            chunk_size=args.chunk_size,
            min_chunk_size=max(80, args.chunk_size // 3),
            similarity_threshold=args.semantic_threshold,
            max_sentences_per_chunk=args.max_sentences,
        )
    return RecursiveChunker(chunk_size=args.chunk_size)


def build_embedder(args: argparse.Namespace):
    if args.embedding == "local":
        try:
            embedder = LocalEmbedder(model_name=args.local_model)
            return embedder, getattr(embedder, "_backend_name", "local")
        except Exception as exc:
            print(f"[WARN] Local embedder unavailable ({exc}). Falling back to mock.")
            return _mock_embed, getattr(_mock_embed, "_backend_name", "mock")

    if args.embedding == "openai":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            print("[WARN] OpenAI embedder unavailable (missing OPENAI_API_KEY). Falling back to mock.")
            return _mock_embed, "mock embeddings fallback"
        try:
            embedder = OpenAIEmbedder(model_name=args.openai_model, api_key=api_key)
            return embedder, getattr(embedder, "_backend_name", "openai")
        except Exception as exc:
            print(f"[WARN] OpenAI embedder unavailable ({exc}). Falling back to mock.")
            return _mock_embed, getattr(_mock_embed, "_backend_name", "mock")

    return _mock_embed, getattr(_mock_embed, "_backend_name", "mock")


def load_corpus_paths(root: Path, custom_docs: list[str] | None) -> list[Path]:
    data_dir = root / "data"
    if custom_docs:
        paths = [Path(p) if Path(p).is_absolute() else (root / p) for p in custom_docs]
    else:
        paths = [
            data_dir / "luatbienvietnam.md",
            data_dir / "nghidinh144.md",
            data_dir / "nghidinhvevantai.md",
            data_dir / "nghidinhvedieukienkinhdoanh.md",
            data_dir / "thongtu66.md",
        ]
    return [p for p in paths if p.exists() and p.is_file()]


def chunk_documents(paths: list[Path], chunker) -> tuple[list[Document], float]:
    docs: list[Document] = []
    chunk_lengths: list[int] = []

    for path in paths:
        text = path.read_text(encoding="utf-8")
        base_metadata = {
            "source_file": path.name,
            "source_path": str(path),
            "doc_type": infer_doc_type(path.name),
            "language": "vi",
        }

        if hasattr(chunker, "chunk_with_metadata"):
            entries = chunker.chunk_with_metadata(text, source=path.name, base_metadata=base_metadata)
            for i, entry in enumerate(entries):
                content = entry.get("content", "")
                metadata = dict(base_metadata)
                metadata.update(entry.get("metadata", {}))
                chunk_lengths.append(len(content))
                docs.append(Document(id=f"{path.stem}_{i}", content=content, metadata=metadata))
        else:
            chunks = chunker.chunk(text)
            for i, chunk in enumerate(chunks):
                chunk_lengths.append(len(chunk))
                docs.append(
                    Document(
                        id=f"{path.stem}_{i}",
                        content=chunk,
                        metadata=base_metadata,
                    )
                )

    avg_chunk_len = mean(chunk_lengths) if chunk_lengths else 0.0
    return docs, avg_chunk_len


def preview(text: str, limit: int = 140) -> str:
    one_line = " ".join(text.split())
    return one_line[:limit] + ("..." if len(one_line) > limit else "")


def run_query(store: EmbeddingStore, item: BenchmarkItem, top_k: int) -> dict:
    if item.requires_filter:
        results = store.search_with_filter(
            item.question,
            top_k=top_k,
            metadata_filter={"doc_type": "decree"},
        )
    else:
        results = store.search(item.question, top_k=top_k)

    top_source = ""
    top_score = 0.0
    if results:
        top_source = results[0].get("metadata", {}).get("source_file", "")
        top_score = float(results[0].get("score", 0.0))

    return {
        "results": results,
        "hit_at_1": bool(results) and expected_match(item.expected_sources, top_source),
        "hit_at_k": any(
            expected_match(item.expected_sources, r.get("metadata", {}).get("source_file", ""))
            for r in results
        ),
        "top_score": top_score,
    }


def print_query_result(item: BenchmarkItem, result: dict, top_k: int, preview_chars: int, query_sec: float) -> None:
    print(f"\n[{item.qid}] {item.question}")
    if item.requires_filter:
        print("filter: doc_type=decree")

    rows = result["results"]
    if not rows:
        print("No results.")
        return

    for idx, row in enumerate(rows, 1):
        src = row.get("metadata", {}).get("source_file", "unknown")
        score = float(row.get("score", 0.0))
        print(f"Top-{idx} | score={score:.4f} | source={src}")
        print(f"  chunk: {preview(row.get('content', ''), preview_chars)}")

    if item.qid != "CUSTOM":
        print(f"hit@1={result['hit_at_1']} | hit@{top_k}={result['hit_at_k']} | query_seconds={query_sec:.4f}")
    else:
        print(f"query_seconds={query_sec:.4f}")


def interactive_loop(store: EmbeddingStore, top_k: int, preview_chars: int) -> None:
    print("\n=== INTERACTIVE MODE ===")
    print("Nhap cau hoi de truy xuat. Go 'exit' de thoat.")
    print("Neu muon bat bo loc nghi dinh, dung cu phap: /decree cau hoi cua ban")

    while True:
        try:
            raw = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped interactive mode.")
            break

        if not raw:
            continue

        if raw.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        requires_filter = False
        question = raw
        if raw.lower().startswith("/decree "):
            requires_filter = True
            question = raw[len("/decree ") :].strip()

        item = BenchmarkItem(
            qid="CUSTOM",
            question=question,
            gold_answer="",
            expected_sources=[],
            requires_filter=requires_filter,
        )

        qs = time.perf_counter()
        result = run_query(store, item, top_k=top_k)
        query_sec = time.perf_counter() - qs
        print_query_result(item, result, top_k=top_k, preview_chars=preview_chars, query_sec=query_sec)


def evaluate_benchmark(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=root / ".env", override=False)
    benchmark_path = root / "data" / "BENCHMARK.md"

    if not benchmark_path.exists():
        print("[ERROR] Missing benchmark file: data/BENCHMARK.md")
        return 1

    benchmark_items = parse_benchmark_md(benchmark_path)
    if not benchmark_items:
        print("[ERROR] Could not parse benchmark rows from data/BENCHMARK.md")
        return 1

    if args.ask:
        benchmark_items = [
            BenchmarkItem(
                qid="CUSTOM",
                question=args.ask,
                gold_answer="",
                expected_sources=[],
                requires_filter=False,
            )
        ]

    corpus_paths = load_corpus_paths(root=root, custom_docs=args.docs)
    if not corpus_paths:
        print("[ERROR] No corpus docs found. Check --docs values.")
        return 1

    chunker = build_chunker(args)
    embedder, backend_name = build_embedder(args)

    t0 = time.perf_counter()
    docs, avg_chunk_len = chunk_documents(corpus_paths, chunker)
    store = EmbeddingStore(collection_name=f"main_new_{int(time.time())}", embedding_fn=embedder)
    store.add_documents(docs)
    build_sec = time.perf_counter() - t0

    print("\n=== RUN CONFIG ===")
    print(f"strategy={args.strategy}")
    if args.strategy == "fixed":
        print(f"chunk_size={args.chunk_size}, overlap={args.overlap}")
    elif args.strategy == "sentence":
        print(f"max_sentences_per_chunk={args.max_sentences}")
    elif args.strategy == "semantic_metadata":
        print(
            f"chunk_size={args.chunk_size}, max_sentences_per_chunk={args.max_sentences}, "
            f"semantic_threshold={args.semantic_threshold}"
        )
    else:
        print(f"chunk_size={args.chunk_size}")
    print(f"embedding_backend={backend_name}")
    print(f"top_k={args.top_k}")

    print("\n=== INDEX STATS ===")
    print(f"documents={len(corpus_paths)}")
    print(f"chunks={len(docs)}")
    print(f"avg_chunk_length={avg_chunk_len:.2f}")
    print(f"build_seconds={build_sec:.4f}")

    if args.interactive:
        interactive_loop(store, top_k=args.top_k, preview_chars=args.preview_chars)
        return 0

    query_times = []
    hit1_count = 0
    hitk_count = 0
    top_scores = []

    print("\n=== QUERY OUTPUT ===")
    for item in benchmark_items:
        qs = time.perf_counter()
        result = run_query(store, item, args.top_k)
        query_sec = time.perf_counter() - qs
        query_times.append(query_sec)
        top_scores.append(result["top_score"])

        if item.qid != "CUSTOM":
            hit1_count += 1 if result["hit_at_1"] else 0
            hitk_count += 1 if result["hit_at_k"] else 0

        print_query_result(
            item,
            result,
            top_k=args.top_k,
            preview_chars=args.preview_chars,
            query_sec=query_sec,
        )

    if benchmark_items and benchmark_items[0].qid != "CUSTOM":
        total = len(benchmark_items)
        avg_q = mean(query_times) if query_times else 0.0
        avg_score = mean(top_scores) if top_scores else 0.0

        print("\n=== SUMMARY ===")
        print(f"top1_source_accuracy={hit1_count}/{total} ({(hit1_count / total) * 100:.1f}%)")
        print(f"hit@{args.top_k}={hitk_count}/{total} ({(hitk_count / total) * 100:.1f}%)")
        print(f"avg_top1_score={avg_score:.4f}")
        print(f"total_query_seconds={sum(query_times):.4f}")
        print(f"avg_query_seconds={avg_q:.4f}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark queries and retrieve top-k chunks from your legal corpus."
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "sentence", "recursive", "semantic_metadata"],
        default="fixed",
    )
    parser.add_argument("--chunk-size", type=int, default=650)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--max-sentences", type=int, default=2)
    parser.add_argument("--semantic-threshold", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--preview-chars", type=int, default=140)
    parser.add_argument("--interactive", action="store_true", help="Interactive question mode.")
    parser.add_argument("--ask", default=None, help="Run one custom question instead of all benchmark rows.")
    parser.add_argument("--docs", nargs="*", default=None, help="Optional custom corpus file paths.")
    parser.add_argument("--embedding", choices=["mock", "local", "openai"], default="local")
    parser.add_argument("--local-model", default=LOCAL_EMBEDDING_MODEL)
    parser.add_argument("--openai-model", default=OPENAI_EMBEDDING_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(evaluate_benchmark(parse_args()))
