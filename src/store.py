from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            client = chromadb.EphemeralClient()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata or {})
        metadata["doc_id"] = doc.id
        embedding = self._embedding_fn(doc.content)
        record_id = f"{doc.id}_{self._next_index}"
        return {"id": record_id, "content": doc.content, "metadata": metadata, "embedding": embedding}

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records or top_k <= 0:
            return []
        query_emb = self._embedding_fn(query)
        scored = []
        for r in records:
            score = _dot(query_emb, r["embedding"])
            scored.append({"id": r.get("id"), "content": r["content"], "metadata": r["metadata"], "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        ids, documents, embeddings, metadatas = [], [], [], []
        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)
            self._next_index += 1
            ids.append(record["id"])
            documents.append(record["content"])
            embeddings.append(record["embedding"])
            metadatas.append(record["metadata"])
        if self._use_chroma and self._collection is not None and ids:
            try:
                self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            except Exception:
                pass

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                return len(self._store)
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self._search_records(query, self._store, top_k)
        filtered = [
            r for r in self._store
            if all(r.get("metadata", {}).get(k) == v for k, v in metadata_filter.items())
        ]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        before = len(self._store)
        keep = [r for r in self._store if r.get("metadata", {}).get("doc_id") != doc_id]
        removed = len(keep) < before
        self._store = keep
        if self._use_chroma and self._collection is not None and removed:
            try:
                existing = self._collection.get()
                del_ids = [i for i, m in zip(existing.get("ids", []), existing.get("metadatas", [])) if (m or {}).get("doc_id") == doc_id]
                if del_ids:
                    self._collection.delete(ids=del_ids)
            except Exception:
                pass
        return removed
