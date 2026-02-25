"""Document ingestion pipelines for parsing and indexing financial documents.

This module provides parsers that convert raw financial documents (SEC filing
HTML, earnings call transcripts) into structured LlamaIndex nodes suitable
for vector indexing, and an index manager for persisting and querying those
nodes via ChromaDB.

Primary classes:

* :class:`SecFilingParser` -- Parses SEC filing HTML into section-tagged nodes.
* :class:`TranscriptParser` -- Parses earnings call transcripts into
  speaker-attributed segments.
* :class:`IndexManager` -- Manages ChromaDB-backed vector indices for
  semantic search over financial documents.
"""

from hermes.ingestion.index_manager import IndexManager
from hermes.ingestion.sec_parser import SecFilingParser
from hermes.ingestion.transcript_parser import TranscriptParser

__all__ = ["IndexManager", "SecFilingParser", "TranscriptParser"]
