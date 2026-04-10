from __future__ import annotations

import math
import re


def _normalize_for_tokens(text: str) -> str:
    lowered = text.lower().replace("đ", "d")
    return re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)


def _sentence_tokens(text: str) -> set[str]:
    tokens = [token for token in _normalize_for_tokens(text).split() if len(token) >= 3]
    return set(tokens)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = _sentence_tokens(text_a)
    tokens_b = _sentence_tokens(text_b)
    if not tokens_a and not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return (intersection / union) if union else 0.0


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split right after sentence-ending punctuation followed by whitespace/newline.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])(?:\s+)", text.strip()) if s.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group).strip())
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        separators = self.separators or [""]
        chunks = self._split(text, separators)
        return [c.strip() for c in chunks if c and c.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Final fallback: hard split by character width.
        if separator == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        pieces = current_text.split(separator)
        if len(pieces) == 1:
            return self._split(current_text, next_separators)

        results: list[str] = []
        buffer = ""

        for piece in pieces:
            candidate = piece if not buffer else f"{buffer}{separator}{piece}"
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            if buffer:
                results.extend(self._split(buffer, next_separators))
                buffer = piece
            else:
                results.extend(self._split(piece, next_separators))
                buffer = ""

        if buffer:
            results.extend(self._split(buffer, next_separators))
        return results


class SemanticMetadataChunker:
    """
    Chunk text by section boundaries and sentence-level semantic continuity.

    Behavior:
        - Prefer splitting at legal/heading boundaries (e.g., "Chuong", "Dieu", markdown headings).
        - Inside each section, group semantically-close neighboring sentences.
        - Emit metadata for each chunk so retrieval can be filtered/analyzed later.
    """

    HEADING_PATTERN = re.compile(
        r"^\s*(#{1,6}\s+.+|chuong\s+[ivxlcdm0-9]+.*|dieu\s+\d+.*|article\s+\d+.*|section\s+\d+.*)$",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])(?:\s+)")

    def __init__(
        self,
        chunk_size: int = 650,
        min_chunk_size: int = 220,
        similarity_threshold: float = 0.15,
        max_sentences_per_chunk: int = 6,
    ) -> None:
        self.chunk_size = max(150, chunk_size)
        self.min_chunk_size = max(80, min_chunk_size)
        self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        if not text.strip():
            return []

        sections: list[tuple[str, str]] = []
        current_title = "preamble"
        current_lines: list[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if self.HEADING_PATTERN.match(line):
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                    current_lines = []
                current_title = line[:120]
                continue
            current_lines.append(raw_line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))

        return [(title, body) for title, body in sections if body]

    def _split_sentences(self, text: str) -> list[str]:
        sentences = [s.strip() for s in self.SENTENCE_SPLIT_PATTERN.split(text.strip()) if s.strip()]
        if sentences:
            return sentences
        return [text.strip()] if text.strip() else []

    def chunk(self, text: str) -> list[str]:
        return [item["content"] for item in self.chunk_with_metadata(text)]

    def chunk_with_metadata(
        self,
        text: str,
        source: str | None = None,
        base_metadata: dict | None = None,
    ) -> list[dict]:
        if not text or not text.strip():
            return []

        base = dict(base_metadata or {})
        chunks: list[dict] = []
        section_index = 0

        for section_title, section_body in self._split_sections(text):
            section_index += 1
            sentences = self._split_sentences(section_body)
            if not sentences:
                continue

            buffer_sentences: list[str] = []
            buffer_len = 0
            chunk_local_index = 0
            start_sentence_idx = 0

            for idx, sentence in enumerate(sentences):
                sentence_len = len(sentence)
                if not buffer_sentences:
                    buffer_sentences = [sentence]
                    buffer_len = sentence_len
                    start_sentence_idx = idx
                    continue

                similarity = _jaccard_similarity(buffer_sentences[-1], sentence)
                candidate_len = buffer_len + 1 + sentence_len

                should_split = False
                if candidate_len > self.chunk_size:
                    should_split = True
                elif len(buffer_sentences) >= self.max_sentences_per_chunk:
                    should_split = True
                elif similarity < self.similarity_threshold and buffer_len >= self.min_chunk_size:
                    should_split = True

                if should_split:
                    chunk_local_index += 1
                    content = " ".join(buffer_sentences).strip()
                    metadata = dict(base)
                    metadata.update(
                        {
                            "source": source,
                            "section_index": section_index,
                            "section_title": section_title,
                            "chunk_in_section": chunk_local_index,
                            "sentence_start": start_sentence_idx,
                            "sentence_end": idx - 1,
                            "semantic_mode": "lexical_jaccard",
                        }
                    )
                    chunks.append({"content": content, "metadata": metadata})

                    buffer_sentences = [sentence]
                    buffer_len = sentence_len
                    start_sentence_idx = idx
                else:
                    buffer_sentences.append(sentence)
                    buffer_len = candidate_len

            if buffer_sentences:
                chunk_local_index += 1
                content = " ".join(buffer_sentences).strip()
                metadata = dict(base)
                metadata.update(
                    {
                        "source": source,
                        "section_index": section_index,
                        "section_title": section_title,
                        "chunk_in_section": chunk_local_index,
                        "sentence_start": start_sentence_idx,
                        "sentence_end": len(sentences) - 1,
                        "semantic_mode": "lexical_jaccard",
                    }
                )
                chunks.append({"content": content, "metadata": metadata})

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        overlap = max(0, min(50, chunk_size // 5, max(0, chunk_size - 1)))

        fixed_chunks = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap).chunk(text)
        sentence_chunks = SentenceChunker(max_sentences_per_chunk=2).chunk(text)
        recursive_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def _stats(chunks: list[str]) -> dict[str, float | int | list[str]]:
            count = len(chunks)
            avg_length = (sum(len(c) for c in chunks) / count) if count else 0.0
            return {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks,
            }

        return {
            "fixed_size": _stats(fixed_chunks),
            "by_sentences": _stats(sentence_chunks),
            "recursive": _stats(recursive_chunks),
        }
