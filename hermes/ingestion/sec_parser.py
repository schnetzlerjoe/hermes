"""SEC filing HTML parser.

Converts raw SEC filing HTML into LlamaIndex :class:`TextNode` instances with
section-level metadata (MD&A, Risk Factors, financial tables, etc.).  The
parser identifies standard filing sections by heading patterns defined in
SEC Regulation S-K and creates separate nodes for each section to enable
targeted retrieval.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from bs4 import BeautifulSoup, Tag
from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)


class SecFilingParser:
    """Parse SEC filing HTML into structured document nodes.

    Identifies standard filing sections by heading patterns (Item 1, Item 1A,
    Item 7, etc.) and creates separate :class:`TextNode` instances for each
    section with metadata tags for downstream filtering.

    The parser handles the wide variety of HTML formatting found in SEC EDGAR
    filings, including nested tables, inconsistent heading styles, and
    non-standard whitespace.

    Example::

        parser = SecFilingParser()
        nodes = parser.parse(html_content, metadata={
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2024-11-01",
            "accession_number": "0000320193-24-000123",
        })
        for node in nodes:
            print(node.metadata["section_name"], len(node.text))
    """

    # Mapping of normalised section identifiers to human-readable names.
    # Keys are lowercased patterns matched against extracted heading text.
    SECTION_PATTERNS: dict[str, str] = {
        "item 1a": "Risk Factors",
        "item 1b": "Unresolved Staff Comments",
        "item 1c": "Cybersecurity",
        "item 1": "Business",
        "item 2": "Properties",
        "item 3": "Legal Proceedings",
        "item 4": "Mine Safety Disclosures",
        "item 5": "Market for Registrant's Common Equity",
        "item 6": "Reserved",
        "item 7a": "Quantitative and Qualitative Disclosures About Market Risk",
        "item 7": "Management's Discussion and Analysis (MD&A)",
        "item 8": "Financial Statements and Supplementary Data",
        "item 9a": "Controls and Procedures",
        "item 9b": "Other Information",
        "item 9": "Changes in and Disagreements with Accountants",
        "item 10": "Directors, Executive Officers and Corporate Governance",
        "item 11": "Executive Compensation",
        "item 12": "Security Ownership",
        "item 13": "Certain Relationships and Related Transactions",
        "item 14": "Principal Accountant Fees and Services",
        "item 15": "Exhibits and Financial Statement Schedules",
        "item 16": "Form 10-K Summary",
        # 10-Q specific items
        "part i": "Financial Information",
        "part ii": "Other Information",
    }

    # Regex for matching "Item N" or "Item NA" headings with optional
    # punctuation, whitespace, and section titles.
    _ITEM_PATTERN: re.Pattern[str] = re.compile(
        r"^\s*item\s+(\d+[a-c]?)\b",
        re.IGNORECASE,
    )

    _PART_PATTERN: re.Pattern[str] = re.compile(
        r"^\s*part\s+(i{1,3}|iv)\b",
        re.IGNORECASE,
    )

    # Minimum character count for a section to be considered substantive
    # and worth indexing.  Filters out empty sections and TOC entries.
    MIN_SECTION_LENGTH: int = 200

    # Maximum character count per node.  Sections exceeding this are split
    # into multiple nodes to stay within embedding model context limits.
    MAX_NODE_LENGTH: int = 8000

    def parse(
        self,
        html: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[TextNode]:
        """Parse filing HTML into text nodes with section metadata.

        Extracts visible text from the HTML, identifies section boundaries
        using heading patterns, and creates one :class:`TextNode` per section.
        Large sections are split into multiple nodes at paragraph boundaries.

        Args:
            html: Raw HTML content of the SEC filing.
            metadata: Optional base metadata dict to merge into every node.
                Typical keys: ``ticker``, ``filing_type``, ``filing_date``,
                ``accession_number``.

        Returns:
            A list of :class:`TextNode` instances, each representing a
            section (or sub-section) of the filing with metadata including
            ``section_id`` and ``section_name``.
        """
        base_meta = metadata or {}
        soup = BeautifulSoup(html, "html.parser")

        # Remove script, style, and hidden elements.
        for tag in soup.find_all(["script", "style", "meta", "link"]):
            tag.decompose()

        # Extract all text blocks, preserving rough document order.
        text_blocks = self._extract_text_blocks(soup)

        # Split text blocks into sections by heading patterns.
        sections = self._split_into_sections(text_blocks)

        # Convert sections into TextNode instances.
        nodes: list[TextNode] = []
        for section_id, section_name, section_text in sections:
            clean_text = self._clean_text(section_text)
            if len(clean_text) < self.MIN_SECTION_LENGTH:
                continue

            node_meta = {
                **base_meta,
                "section_id": section_id,
                "section_name": section_name,
            }

            # Split large sections into sub-nodes.
            chunks = self._split_long_text(clean_text)
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
            "Parsed SEC filing into %d nodes across %d sections",
            len(nodes),
            len(sections),
        )
        return nodes

    def parse_tables(self, html: str) -> list[dict[str, Any]]:
        """Extract HTML tables as structured data.

        Each table is returned as a dictionary with keys ``headers`` (list of
        column header strings) and ``rows`` (list of lists of cell strings).
        Tables with fewer than 2 rows are skipped as they are typically
        formatting artifacts rather than data tables.

        Args:
            html: Raw HTML content containing tables.

        Returns:
            A list of dicts, each with ``headers`` and ``rows`` keys.
        """
        soup = BeautifulSoup(html, "html.parser")
        tables: list[dict[str, Any]] = []

        for table_tag in soup.find_all("table"):
            rows = table_tag.find_all("tr")
            if len(rows) < 2:
                continue

            parsed_rows: list[list[str]] = []
            for row in rows:
                cells = row.find_all(["td", "th"])
                parsed_rows.append(
                    [self._clean_text(cell.get_text()) for cell in cells]
                )

            if not parsed_rows:
                continue

            # Use the first row as headers if it contains <th> elements
            # or looks like a header row (all non-numeric).
            first_row = rows[0]
            has_th = first_row.find("th") is not None
            if has_th and len(parsed_rows) > 1:
                headers = parsed_rows[0]
                data_rows = parsed_rows[1:]
            else:
                headers = [f"col_{i}" for i in range(len(parsed_rows[0]))]
                data_rows = parsed_rows

            tables.append({"headers": headers, "rows": data_rows})

        logger.debug("Extracted %d tables from HTML", len(tables))
        return tables

    def _identify_section(self, text: str) -> tuple[str, str] | None:
        """Match text against known SEC filing section patterns.

        Attempts to match the given text against Item and Part heading
        patterns.  Returns the normalised section identifier and human-
        readable name, or ``None`` if no match is found.

        Args:
            text: A text string, typically a heading extracted from the filing.

        Returns:
            A ``(section_id, section_name)`` tuple if the text matches a
            known heading pattern, otherwise ``None``.
        """
        stripped = text.strip()

        # Try Item pattern first (more specific).
        item_match = self._ITEM_PATTERN.match(stripped)
        if item_match:
            item_key = f"item {item_match.group(1).lower()}"
            section_name = self.SECTION_PATTERNS.get(item_key)
            if section_name:
                return item_key, section_name

        # Try Part pattern.
        part_match = self._PART_PATTERN.match(stripped)
        if part_match:
            part_key = f"part {part_match.group(1).lower()}"
            section_name = self.SECTION_PATTERNS.get(part_key)
            if section_name:
                return part_key, section_name

        return None

    def _extract_text_blocks(self, soup: BeautifulSoup) -> list[str]:
        """Extract visible text blocks from parsed HTML in document order.

        Walks the DOM and extracts text from block-level elements (p, div,
        headings, table cells, spans) while preserving the document reading
        order.

        Args:
            soup: A parsed BeautifulSoup document.

        Returns:
            A list of non-empty text strings in document order.
        """
        blocks: list[str] = []
        block_tags = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                       "li", "td", "th", "span", "font", "b", "i", "u"}

        for element in soup.descendants:
            if isinstance(element, Tag) and element.name in block_tags:
                # Only get direct text to avoid duplication from nested tags.
                direct_text = element.get_text(separator=" ", strip=True)
                if direct_text and len(direct_text) > 1:
                    blocks.append(direct_text)

        # Deduplicate consecutive identical blocks (common in nested HTML).
        deduped: list[str] = []
        prev = ""
        for block in blocks:
            if block != prev:
                deduped.append(block)
                prev = block

        return deduped

    def _split_into_sections(
        self,
        text_blocks: list[str],
    ) -> list[tuple[str, str, str]]:
        """Split text blocks into sections based on heading patterns.

        Scans through text blocks looking for section headings.  Text between
        headings is assigned to the preceding section.  Text before the first
        heading is assigned to a ``preamble`` section.

        Args:
            text_blocks: Ordered list of text strings from the filing.

        Returns:
            A list of ``(section_id, section_name, section_text)`` tuples.
        """
        sections: list[tuple[str, str, str]] = []
        current_id = "preamble"
        current_name = "Preamble"
        current_text_parts: list[str] = []

        for block in text_blocks:
            match = self._identify_section(block)
            if match is not None:
                # Save the current section before starting a new one.
                if current_text_parts:
                    full_text = "\n\n".join(current_text_parts)
                    sections.append((current_id, current_name, full_text))

                current_id, current_name = match
                current_text_parts = []
            else:
                current_text_parts.append(block)

        # Don't forget the last section.
        if current_text_parts:
            full_text = "\n\n".join(current_text_parts)
            sections.append((current_id, current_name, full_text))

        return sections

    def _split_long_text(self, text: str) -> list[str]:
        """Split text exceeding ``MAX_NODE_LENGTH`` at paragraph boundaries.

        Attempts to split at double newlines first, then at single newlines,
        and finally by hard truncation as a last resort.

        Args:
            text: The text to potentially split.

        Returns:
            A list of one or more text chunks, each within the length limit.
        """
        if len(text) <= self.MAX_NODE_LENGTH:
            return [text]

        chunks: list[str] = []
        paragraphs = text.split("\n\n")
        current_chunk: list[str] = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para) + 2  # Account for the join separator.
            if current_length + para_len > self.MAX_NODE_LENGTH and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Handle individual paragraphs that exceed the limit.
            if para_len > self.MAX_NODE_LENGTH:
                # Hard split at MAX_NODE_LENGTH boundaries.
                for i in range(0, len(para), self.MAX_NODE_LENGTH):
                    chunks.append(para[i : i + self.MAX_NODE_LENGTH])
            else:
                current_chunk.append(para)
                current_length += para_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise whitespace and strip control characters.

        Collapses multiple spaces/tabs into single spaces, normalises
        line endings, and removes leading/trailing whitespace.

        Args:
            text: Raw text string.

        Returns:
            Cleaned text string.
        """
        # Replace various whitespace characters with spaces.
        text = re.sub(r"[\t\r\xa0]+", " ", text)
        # Collapse multiple spaces.
        text = re.sub(r" {2,}", " ", text)
        # Collapse more than two consecutive newlines.
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
