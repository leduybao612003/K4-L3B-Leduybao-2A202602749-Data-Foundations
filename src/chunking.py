from __future__ import annotations

import math
import re


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
        if not text or not text.strip():
            return []
        stripped = text.strip()
        # Split on ". ", "! ", "? " or ".\n" (any sentence-ending punct + whitespace)
        sentences = re.split(r"(?<=[.!?])\s+", stripped)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [stripped]
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
        if len(text) <= self.chunk_size:
            return [text]
        if not self.separators:
            # No separators: fall back to fixed-size slicing.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]
        sep = remaining_separators[0]
        rest = remaining_separators[1:]
        if sep == "":
            # Character-level fallback: slice to chunk_size.
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]
        if sep not in current_text:
            return self._split(current_text, rest)
        parts = current_text.split(sep)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = part if current == "" else current + sep + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.extend(self._split(current, rest))
                # part itself may still be too large -> recurse
                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part, rest))
                    current = ""
                else:
                    current = part
        if current:
            chunks.extend(self._split(current, rest))
        return [c for c in chunks if c]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed = FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(text)
        by_sent = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        rec = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg = sum(len(c) for c in chunks) / count if count else 0.0
            return {"count": count, "avg_length": avg, "chunks": chunks}

        return {
            "fixed_size": stats(fixed),
            "by_sentences": stats(by_sent),
            "recursive": stats(rec),
        }
