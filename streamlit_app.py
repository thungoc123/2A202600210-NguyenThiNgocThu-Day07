from __future__ import annotations

import os
import time
import unicodedata
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import streamlit as st
from dotenv import load_dotenv

from src.chunking import (
    ChunkingStrategyComparator,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticMetadataChunker,
    SentenceChunker,
)
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
    if not md_path.exists():
        return []

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


def build_chunker(
    strategy: str,
    chunk_size: int,
    overlap: int,
    max_sentences: int,
    semantic_threshold: float,
):
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "sentence":
        return SentenceChunker(max_sentences_per_chunk=max_sentences)
    if strategy == "semantic_metadata":
        return SemanticMetadataChunker(
            chunk_size=chunk_size,
            min_chunk_size=max(80, chunk_size // 3),
            similarity_threshold=semantic_threshold,
            max_sentences_per_chunk=max_sentences,
        )
    return RecursiveChunker(chunk_size=chunk_size)


def build_embedder(embedding: str, local_model: str, openai_model: str):
    if embedding == "local":
        try:
            embedder = LocalEmbedder(model_name=local_model)
            return embedder, getattr(embedder, "_backend_name", "local"), ""
        except Exception as exc:
            return _mock_embed, getattr(_mock_embed, "_backend_name", "mock"), f"Local embedder fallback: {exc}"

    if embedding == "openai":
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            return (
                _mock_embed,
                getattr(_mock_embed, "_backend_name", "mock"),
                "OpenAI embedder fallback: missing OPENAI_API_KEY. Add it to .env or environment variables.",
            )
        try:
            embedder = OpenAIEmbedder(model_name=openai_model, api_key=api_key)
            return embedder, getattr(embedder, "_backend_name", "openai"), ""
        except Exception as exc:
            return _mock_embed, getattr(_mock_embed, "_backend_name", "mock"), f"OpenAI embedder fallback: {exc}"

    return _mock_embed, getattr(_mock_embed, "_backend_name", "mock"), ""


def preview(text: str, limit: int = 220) -> str:
    one_line = " ".join((text or "").split())
    return one_line[:limit] + ("..." if len(one_line) > limit else "")


@st.cache_resource(show_spinner=False)
def build_store_cached(
    root_dir: str,
    selected_docs: tuple[str, ...],
    strategy: str,
    chunk_size: int,
    overlap: int,
    max_sentences: int,
    semantic_threshold: float,
    embedding: str,
    local_model: str,
    openai_model: str,
):
    root = Path(root_dir)
    data_dir = root / "data"

    chunker = build_chunker(strategy, chunk_size, overlap, max_sentences, semantic_threshold)
    embedder, backend_name, warning = build_embedder(embedding, local_model, openai_model)

    docs: list[Document] = []
    chunk_lengths: list[int] = []
    raw_chars = 0

    for file_name in selected_docs:
        path = data_dir / file_name
        text = path.read_text(encoding="utf-8")
        raw_chars += len(text)
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

    build_start = time.perf_counter()
    store = EmbeddingStore(collection_name=f"streamlit_{int(time.time())}", embedding_fn=embedder)
    store.add_documents(docs)
    build_seconds = time.perf_counter() - build_start

    stats = {
        "backend": backend_name,
        "warning": warning,
        "documents": len(selected_docs),
        "raw_chars": raw_chars,
        "chunks": len(docs),
        "avg_chunk_length": mean(chunk_lengths) if chunk_lengths else 0.0,
        "build_seconds": build_seconds,
    }
    return store, stats


def run_custom_query(store: EmbeddingStore, question: str, top_k: int, use_decree_filter: bool):
    if use_decree_filter:
        return store.search_with_filter(question, top_k=top_k, metadata_filter={"doc_type": "decree"})
    return store.search(question, top_k=top_k)


def gold_keyword_coverage(gold_answer: str, chunk_text: str) -> float:
    gold_tokens = [token for token in normalize_text(gold_answer).split() if len(token) >= 5]
    if not gold_tokens:
        return 0.0

    unique_tokens = sorted(set(gold_tokens))
    chunk_norm = normalize_text(chunk_text)
    overlap = sum(1 for token in unique_tokens if token in chunk_norm)
    return overlap / len(unique_tokens)


def run_benchmark_suite(store: EmbeddingStore, items: list[BenchmarkItem], top_k: int):
    rows = []
    query_times = []

    hit1_count = 0
    hitk_count = 0

    for item in items:
        start = time.perf_counter()

        if item.requires_filter:
            results = store.search_with_filter(item.question, top_k=top_k, metadata_filter={"doc_type": "decree"})
        else:
            results = store.search(item.question, top_k=top_k)

        query_seconds = time.perf_counter() - start
        query_times.append(query_seconds)

        top_source = results[0].get("metadata", {}).get("source_file", "") if results else ""
        top_score = float(results[0].get("score", 0.0)) if results else 0.0
        top_chunk = results[0].get("content", "") if results else ""

        hit_at_1 = bool(results) and expected_match(item.expected_sources, top_source)
        hit_at_k = any(
            expected_match(item.expected_sources, row.get("metadata", {}).get("source_file", ""))
            for row in results
        )

        if hit_at_1:
            hit1_count += 1
        if hit_at_k:
            hitk_count += 1

        rows.append(
            {
                "qid": item.qid,
                "hit@1": hit_at_1,
                f"hit@{top_k}": hit_at_k,
                "top1_score": round(top_score, 4),
                "top1_source": top_source,
                "query_seconds": round(query_seconds, 4),
                "gold_coverage": round(gold_keyword_coverage(item.gold_answer, top_chunk), 4),
            }
        )

    total = len(items) if items else 1
    summary = {
        "top1_source_accuracy": f"{hit1_count}/{len(items)} ({(hit1_count / total) * 100:.1f}%)",
        f"hit@{top_k}": f"{hitk_count}/{len(items)} ({(hitk_count / total) * 100:.1f}%)",
        "avg_query_seconds": round(mean(query_times), 4) if query_times else 0.0,
    }
    return rows, summary


def run_chunking_comparison(data_dir: Path, selected_docs: list[str], chunk_size: int):
    comp = ChunkingStrategyComparator()
    rows = []

    for file_name in selected_docs:
        text = (data_dir / file_name).read_text(encoding="utf-8")
        stats = comp.compare(text, chunk_size=chunk_size)
        for strategy_name in ["fixed_size", "by_sentences", "recursive"]:
            rows.append(
                {
                    "document": file_name,
                    "strategy": strategy_name,
                    "chunk_count": stats[strategy_name]["count"],
                    "avg_length": round(stats[strategy_name]["avg_length"], 2),
                }
            )

    return rows


def main() -> None:
    st.set_page_config(page_title="RAG Strategy Lab", layout="wide")
    st.title("RAG Strategy Test UI")
    st.caption("Test chunking strategies, embeddings, and benchmark quality in one place.")

    root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=root / ".env", override=False)
    data_dir = root / "data"
    benchmark_file = data_dir / "BENCHMARK.md"

    available_docs = sorted([
        p.name for p in data_dir.glob("*.md") if p.name.upper() != "BENCHMARK.MD"
    ] + [p.name for p in data_dir.glob("*.txt")])

    default_docs = [
        name
        for name in [
            "luatbienvietnam.md",
            "nghidinh144.md",
            "nghidinhvevantai.md",
            "nghidinhvedieukienkinhdoanh.md",
            "thongtu66.md",
        ]
        if name in available_docs
    ]
    if not default_docs:
        default_docs = available_docs[:5]

    st.sidebar.header("Configuration")
    strategy = st.sidebar.selectbox(
        "Chunk strategy",
        ["fixed", "sentence", "recursive", "semantic_metadata"],
        index=0,
    )
    chunk_size = st.sidebar.slider("Chunk size", min_value=200, max_value=1200, value=650, step=50)
    overlap = st.sidebar.slider("Overlap", min_value=0, max_value=300, value=120, step=10)
    max_sentences = st.sidebar.slider("Max sentences per chunk", min_value=1, max_value=8, value=2, step=1)
    semantic_threshold = st.sidebar.slider(
        "Semantic threshold",
        min_value=0.00,
        max_value=0.60,
        value=0.15,
        step=0.01,
    )

    embedding = st.sidebar.selectbox("Embedding backend", ["local", "mock", "openai"], index=0)
    local_model = st.sidebar.text_input("Local model", value=LOCAL_EMBEDDING_MODEL)
    openai_model = st.sidebar.text_input("OpenAI model", value=OPENAI_EMBEDDING_MODEL)
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=3, step=1)
    preview_chars = st.sidebar.slider("Preview chars", min_value=80, max_value=500, value=220, step=20)

    selected_docs = st.sidebar.multiselect(
        "Corpus docs",
        options=available_docs,
        default=default_docs,
    )

    if not selected_docs:
        st.error("Please select at least one document in the sidebar.")
        st.stop()

    with st.spinner("Building index..."):
        store, stats = build_store_cached(
            str(root),
            tuple(selected_docs),
            strategy,
            chunk_size,
            overlap,
            max_sentences,
            semantic_threshold,
            embedding,
            local_model,
            openai_model,
        )

    if stats["warning"]:
        st.warning(stats["warning"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Embedding backend", stats["backend"])
    c2.metric("Documents", stats["documents"])
    c3.metric("Chunks", stats["chunks"])
    c4.metric("Build seconds", f"{stats['build_seconds']:.4f}")

    st.caption(
        f"Raw chars: {stats['raw_chars']} | Avg chunk length: {stats['avg_chunk_length']:.2f}"
    )

    tab_query, tab_benchmark, tab_compare = st.tabs(["Custom Query", "Benchmark", "Chunking Compare"])

    with tab_query:
        st.subheader("Ask a custom question")
        question = st.text_area("Question", value="Đảo được định nghĩa như thế nào trong Luật Biển Việt Nam?")
        use_filter = st.checkbox("Apply decree filter (doc_type=decree)", value=False)

        if st.button("Run query", type="primary"):
            if not question.strip():
                st.error("Please enter a question.")
            else:
                start = time.perf_counter()
                results = run_custom_query(store, question, top_k=top_k, use_decree_filter=use_filter)
                elapsed = time.perf_counter() - start
                st.write(f"Query time: {elapsed:.4f}s")

                if not results:
                    st.info("No results found.")
                else:
                    for idx, row in enumerate(results, 1):
                        src = row.get("metadata", {}).get("source_file", "unknown")
                        score = float(row.get("score", 0.0))
                        st.markdown(f"**Top-{idx}** | score={score:.4f} | source={src}")
                        st.code(preview(row.get("content", ""), preview_chars), language="text")

    with tab_benchmark:
        st.subheader("Run benchmark suite")
        items = parse_benchmark_md(benchmark_file)
        if not items:
            st.error("No benchmark rows found in data/BENCHMARK.md")
        else:
            st.write(f"Loaded {len(items)} benchmark questions from data/BENCHMARK.md")
            if st.button("Run benchmark", type="secondary"):
                rows, summary = run_benchmark_suite(store, items, top_k=top_k)
                st.write("Summary")
                st.json(summary)
                st.write("Per-question results")
                st.dataframe(rows, use_container_width=True)

    with tab_compare:
        st.subheader("Compare built-in chunking strategies")
        st.write("This compares fixed_size, by_sentences, and recursive on the selected documents.")
        if st.button("Run chunking compare"):
            rows = run_chunking_comparison(data_dir, selected_docs, chunk_size=chunk_size)
            st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()
