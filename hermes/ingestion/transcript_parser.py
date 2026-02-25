"""Earnings call transcript parser.

Converts raw earnings call transcript text into structured LlamaIndex
:class:`TextNode` instances attributed to individual speakers and tagged
with section metadata (prepared remarks vs. Q&A).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)


class TranscriptParser:
    """Parse earnings call transcripts into structured segments.

    Identifies speakers, separates prepared remarks from Q&A,
    and tags each segment with metadata including speaker name, role,
    affiliation, and section (prepared remarks or Q&A).

    The parser handles common transcript formats from major providers
    (Seeking Alpha, Motley Fool, S&P Capital IQ, Bloomberg) by
    detecting speaker attribution patterns like "John Smith -- CEO"
    or "John Smith - Goldman Sachs - Analyst".

    Example::

        parser = TranscriptParser()
        nodes = parser.parse(transcript_text, metadata={
            "ticker": "AAPL",
            "event_type": "earnings_call",
            "quarter": "Q4 2024",
            "date": "2024-11-01",
        })
        for node in nodes:
            print(node.metadata["speaker"], node.metadata["section"])
    """

    # Regex patterns for speaker identification.
    # Matches patterns like:
    #   "Tim Cook -- Chief Executive Officer"
    #   "Tim Cook - CEO"
    #   "Tim Cook, CEO"
    #   "Analyst: Jane Doe, Goldman Sachs"
    _SPEAKER_DASH_PATTERN: re.Pattern[str] = re.compile(
        r"^(?P<name>[A-Z][a-zA-Z\.\-\' ]{2,40})\s*"
        r"(?:--|---|-|,)\s*"
        r"(?P<role>.+?)\s*$",
        re.MULTILINE,
    )

    # Matches "Operator" standalone line (common in transcripts).
    _OPERATOR_PATTERN: re.Pattern[str] = re.compile(
        r"^\s*Operator\s*$",
        re.MULTILINE,
    )

    # Section boundary markers.
    _QA_SECTION_MARKERS: tuple[str, ...] = (
        "question-and-answer session",
        "question and answer session",
        "questions and answers",
        "q&a session",
        "q & a session",
        "q&a",
        "questions & answers",
        "analyst q&a",
    )

    _PREPARED_REMARKS_MARKERS: tuple[str, ...] = (
        "prepared remarks",
        "opening remarks",
        "presentation",
        "corporate participants",
        "company participants",
    )

    # Minimum text length for a segment to be worth indexing.
    MIN_SEGMENT_LENGTH: int = 50

    # Maximum text length per node before splitting.
    MAX_NODE_LENGTH: int = 6000

    def parse(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextNode]:
        """Parse transcript text into speaker-attributed segments.

        Splits the transcript into segments by speaker, identifies the
        transition from prepared remarks to Q&A, and creates a
        :class:`TextNode` for each segment with appropriate metadata.

        Args:
            text: Full text of the earnings call transcript.
            metadata: Optional base metadata dict merged into every node.
                Typical keys: ``ticker``, ``event_type``, ``quarter``,
                ``date``.

        Returns:
            A list of :class:`TextNode` instances, each representing a
            speaker's segment with metadata including ``speaker``,
            ``speaker_role``, and ``section``.
        """
        base_meta = metadata or {}

        # Determine the boundary between prepared remarks and Q&A.
        qa_start_idx = self._find_qa_boundary(text)

        # Split transcript into speaker segments.
        segments = self._split_by_speaker(text)

        nodes: list[TextNode] = []
        for segment in segments:
            speaker = segment["speaker"]
            role = segment["role"]
            segment_text = segment["text"].strip()
            start_pos = segment["start_pos"]

            if len(segment_text) < self.MIN_SEGMENT_LENGTH:
                continue

            # Determine section based on position relative to Q&A boundary.
            if qa_start_idx is not None and start_pos >= qa_start_idx:
                section = "q_and_a"
            else:
                section = "prepared_remarks"

            node_meta = {
                **base_meta,
                "speaker": speaker,
                "speaker_role": role,
                "section": section,
            }

            # Split long segments into multiple nodes.
            chunks = self._split_long_text(segment_text)
            for i, chunk in enumerate(chunks):
                chunk_meta = {**node_meta}
                if len(chunks) > 1:
                    chunk_meta["chunk_index"] = i
                    chunk_meta["total_chunks"] = len(chunks)

                nodes.append(
                    TextNode(
                        text=chunk,
                        metadata=chunk_meta,
                    )
                )

        logger.info(
            "Parsed transcript into %d nodes (%d speaker segments)",
            len(nodes),
            len(segments),
        )
        return nodes

    def _find_qa_boundary(self, text: str) -> int | None:
        """Find the character position where the Q&A session begins.

        Searches for common Q&A section markers in the transcript text.
        Returns the character index of the first match, or ``None`` if
        no boundary is found.

        Args:
            text: Full transcript text.

        Returns:
            Character index of Q&A section start, or ``None``.
        """
        text_lower = text.lower()
        earliest: int | None = None

        for marker in self._QA_SECTION_MARKERS:
            idx = text_lower.find(marker)
            if idx != -1 and (earliest is None or idx < earliest):
                earliest = idx

        return earliest

    def _split_by_speaker(self, text: str) -> list[dict[str, Any]]:
        """Split transcript text into speaker-attributed segments.

        Identifies speaker lines using regex patterns and splits the text
        at each speaker boundary.  The Operator is treated as a special
        speaker.

        Args:
            text: Full transcript text.

        Returns:
            A list of dicts with keys ``speaker``, ``role``, ``text``,
            and ``start_pos``.
        """
        segments: list[dict[str, Any]] = []

        # Find all speaker boundaries.
        boundaries: list[tuple[int, int, str, str]] = []

        for match in self._SPEAKER_DASH_PATTERN.finditer(text):
            name = match.group("name").strip()
            role = match.group("role").strip()
            boundaries.append((match.start(), match.end(), name, role))

        for match in self._OPERATOR_PATTERN.finditer(text):
            boundaries.append((match.start(), match.end(), "Operator", "Operator"))

        # Sort by position in the document.
        boundaries.sort(key=lambda b: b[0])

        if not boundaries:
            # No speakers identified -- return the entire text as one segment.
            return [
                {
                    "speaker": "Unknown",
                    "role": "Unknown",
                    "text": text,
                    "start_pos": 0,
                }
            ]

        # Handle text before the first speaker (e.g., disclaimer or header).
        first_start = boundaries[0][0]
        if first_start > self.MIN_SEGMENT_LENGTH:
            segments.append(
                {
                    "speaker": "Header",
                    "role": "Preamble",
                    "text": text[:first_start],
                    "start_pos": 0,
                }
            )

        # Create segments between consecutive speaker boundaries.
        for i, (start, end, name, role) in enumerate(boundaries):
            # Text runs from the end of this speaker line to the start of the
            # next speaker line (or end of document).
            if i + 1 < len(boundaries):
                next_start = boundaries[i + 1][0]
                segment_text = text[end:next_start]
            else:
                segment_text = text[end:]

            segments.append(
                {
                    "speaker": name,
                    "role": role,
                    "text": segment_text,
                    "start_pos": start,
                }
            )

        return segments

    def _split_long_text(self, text: str) -> list[str]:
        """Split text exceeding ``MAX_NODE_LENGTH`` at paragraph boundaries.

        Args:
            text: The text to potentially split.

        Returns:
            A list of one or more text chunks within the length limit.
        """
        if len(text) <= self.MAX_NODE_LENGTH:
            return [text]

        chunks: list[str] = []
        paragraphs = text.split("\n\n")
        current_parts: list[str] = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para) + 2
            if current_length + para_len > self.MAX_NODE_LENGTH and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_length = 0

            if para_len > self.MAX_NODE_LENGTH:
                for j in range(0, len(para), self.MAX_NODE_LENGTH):
                    chunks.append(para[j : j + self.MAX_NODE_LENGTH])
            else:
                current_parts.append(para)
                current_length += para_len

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks
