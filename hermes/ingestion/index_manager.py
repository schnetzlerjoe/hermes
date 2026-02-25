"""ChromaDB index creation and management.

Handles creating, loading, and querying per-company vector indices used by
RAG-based agents for semantic search over financial documents.  Each company
gets its own ChromaDB collection (e.g., ``sec_filings_AAPL``) for isolated
retrieval, preventing cross-company contamination in search results.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from hermes.config import get_config

logger = logging.getLogger(__name__)


class IndexManager:
    """Manage ChromaDB-backed vector indices for financial documents.

    Creates one ChromaDB collection per logical grouping (typically per
    company and document type, e.g., ``sec_filings_AAPL``,
    ``transcripts_AAPL``) for isolated retrieval.  Supports adding
    documents, querying, listing indices, and cleanup.

    The ChromaDB client is configured for persistent storage so that
    indices survive process restarts.  The storage location is determined
    by :attr:`HermesConfig.chroma_persist_dir`.

    Example::

        from hermes.ingestion import IndexManager, SecFilingParser

        manager = IndexManager()
        parser = SecFilingParser()

        nodes = parser.parse(filing_html, metadata={"ticker": "AAPL"})
        manager.add_documents("sec_filings_AAPL", nodes)

        results = manager.query("sec_filings_AAPL", "What are the risk factors?")
        for node in results:
            print(node.text[:200])
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        """Initialize with a ChromaDB persistent client.

        Args:
            persist_dir: Directory for ChromaDB persistent storage.  If
                ``None``, uses the ``chroma_persist_dir`` from the global
                :class:`HermesConfig`.
        """
        if persist_dir is None:
            persist_dir = get_config().chroma_persist_dir

        self._persist_dir = persist_dir
        self._client: chromadb.ClientAPI = chromadb.PersistentClient(
            path=persist_dir,
        )
        # Cache of VectorStoreIndex instances keyed by collection name.
        self._index_cache: dict[str, VectorStoreIndex] = {}

        logger.debug("IndexManager initialized with persist_dir=%s", persist_dir)

    @property
    def persist_dir(self) -> str:
        """Return the ChromaDB persistence directory path."""
        return self._persist_dir

    def get_or_create_index(self, collection_name: str) -> VectorStoreIndex:
        """Get an existing index or create a new empty one.

        If the collection already exists in ChromaDB, wraps it in a
        :class:`VectorStoreIndex`.  Otherwise creates a new empty collection
        and returns an index backed by it.

        Args:
            collection_name: The name for the ChromaDB collection.  Use a
                descriptive convention like ``{doc_type}_{ticker}``
                (e.g., ``sec_filings_AAPL``).

        Returns:
            A :class:`VectorStoreIndex` backed by the named ChromaDB
            collection.
        """
        if collection_name in self._index_cache:
            return self._index_cache[collection_name]

        chroma_collection = self._client.get_or_create_collection(
            name=collection_name,
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
        )

        self._index_cache[collection_name] = index
        logger.debug("Index for collection %r ready", collection_name)
        return index

    def add_documents(
        self,
        collection_name: str,
        nodes: list[TextNode],
    ) -> None:
        """Add document nodes to a collection's index.

        Retrieves (or creates) the index for the given collection and
        inserts the provided nodes.  Nodes are embedded using the
        default embedding model configured in LlamaIndex.

        Args:
            collection_name: Target collection name.
            nodes: List of :class:`TextNode` instances to index.  Each node
                should have meaningful ``text`` and ``metadata`` attributes.

        Raises:
            ValueError: If ``nodes`` is empty.
        """
        if not nodes:
            raise ValueError("Cannot add an empty list of nodes.")

        index = self.get_or_create_index(collection_name)
        index.insert_nodes(nodes)

        logger.info(
            "Added %d nodes to collection %r",
            len(nodes),
            collection_name,
        )

    def query(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[TextNode]:
        """Query a collection and return relevant nodes.

        Performs semantic search against the named collection using the
        provided query string.  Returns the top-k most relevant nodes.

        Args:
            collection_name: Collection to query.
            query: Natural-language query string for semantic search.
            top_k: Maximum number of results to return.  Defaults to 5.
            filters: Optional metadata filters to narrow the search.  Keys
                should match metadata fields on the stored nodes (e.g.,
                ``{"section_name": "Risk Factors"}``).

        Returns:
            A list of :class:`TextNode` instances ranked by relevance.
            May return fewer than ``top_k`` if the collection has
            insufficient documents.

        Raises:
            KeyError: If the collection does not exist.
        """
        if collection_name not in {
            c.name for c in self._client.list_collections()
        }:
            raise KeyError(
                f"Collection '{collection_name}' does not exist. "
                f"Available: {self.list_collections()}"
            )

        index = self.get_or_create_index(collection_name)
        retriever = index.as_retriever(similarity_top_k=top_k)

        retrieved_nodes = retriever.retrieve(query)

        results: list[TextNode] = []
        for node_with_score in retrieved_nodes:
            node = node_with_score.node
            if isinstance(node, TextNode):
                # Attach the similarity score as metadata for transparency.
                node.metadata["similarity_score"] = node_with_score.score
                results.append(node)

        logger.debug(
            "Query on %r returned %d results (top_k=%d)",
            collection_name,
            len(results),
            top_k,
        )
        return results

    def list_collections(self) -> list[str]:
        """List all available collection names.

        Returns:
            A sorted list of collection name strings currently stored
            in ChromaDB.
        """
        collections = self._client.list_collections()
        return sorted(c.name for c in collections)

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection and all its data.

        Removes the collection from ChromaDB and evicts it from the
        in-memory index cache.  This operation is irreversible.

        Args:
            collection_name: The collection to delete.

        Raises:
            KeyError: If the collection does not exist.
        """
        existing = {c.name for c in self._client.list_collections()}
        if collection_name not in existing:
            raise KeyError(
                f"Collection '{collection_name}' does not exist. "
                f"Available: {sorted(existing)}"
            )

        self._client.delete_collection(name=collection_name)
        self._index_cache.pop(collection_name, None)

        logger.info("Deleted collection %r", collection_name)

    def collection_count(self, collection_name: str) -> int:
        """Return the number of documents in a collection.

        Args:
            collection_name: The collection to count.

        Returns:
            The number of embedded documents in the collection.

        Raises:
            KeyError: If the collection does not exist.
        """
        existing = {c.name for c in self._client.list_collections()}
        if collection_name not in existing:
            raise KeyError(
                f"Collection '{collection_name}' does not exist. "
                f"Available: {sorted(existing)}"
            )

        chroma_collection = self._client.get_collection(name=collection_name)
        return chroma_collection.count()

    def __repr__(self) -> str:
        collections = self.list_collections()
        return (
            f"IndexManager(persist_dir={self._persist_dir!r}, "
            f"collections={len(collections)})"
        )
