import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PageSegment:
    """
    Continuous piece of source text belonging to one PDF page.
    """

    text: str
    page: int


@dataclass
class ContentBlock:
    """
    Semantic document unit.

    block_type:
        paragraph
        table

    Paragraph:
        segments contain page-aware prose.

    Table:
        table_header contains everything through the Markdown
        separator row, plus any detected continuation header rows.

        table_rows contains only actual body rows.

        table_row_pages contains the exact page corresponding
        positionally to every table_rows entry.

    page_spans:
        Exact character/page provenance relative to final
        paragraph chunk text.
    """

    block_type: str

    segments: List[PageSegment] = field(
        default_factory=list
    )

    heading_path: List[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Table-specific fields
    # --------------------------------------------------------

    table_header: Optional[str] = None

    table_rows: List[str] = field(
        default_factory=list
    )

    table_row_pages: List[int] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Exact character/page provenance.
    #
    # Coordinates are relative to the final chunk text.
    # --------------------------------------------------------

    page_spans: List[Dict] = field(
        default_factory=list
    )

    @property
    def text(self) -> str:
        """
        Return continuous text represented by segments.
        """

        return "\n".join(
            segment.text
            for segment in self.segments
            if segment.text
        ).strip()

    @property
    def pages(self) -> Set[int]:
        """
        Return all pages contributing to this block.
        """

        pages = {
            segment.page
            for segment in self.segments
        }

        pages.update(
            self.table_row_pages
        )

        return pages


# ============================================================
# CHUNKER
# ============================================================

class FinTechMarkdownChunker:
    """
    Markdown-aware, page-aware and token-aware chunker
    designed for financial-document RAG and ChromaDB.

    Guarantees:

    1. Prose can continue across PDF page boundaries.
    2. Prose is split semantically rather than by page.
    3. Page provenance is calculated using character intervals.
    4. Tables are never passed through the prose splitter.
    5. Tables are split by complete rows whenever possible.
    6. Complete table headers are repeated in every table chunk.
    7. Multi-line table headers are detected dynamically.
    8. Every emitted chunk is validated against the token budget.
    9. Heading hierarchy is preserved.
    10. ChromaDB receives serializable metadata.
    11. table_rows[i] always corresponds to table_row_pages[i].
    12. Oversized table rows account for heading + table header
        when calculating their token budget.
    """

    # ========================================================
    # REGEX
    # ========================================================

    PAGE_START_REGEX = re.compile(
        r"^---\s*\[START OF PAGE (\d+)\]\s*---$"
    )

    PAGE_END_REGEX = re.compile(
        r"^---\s*\[END OF PAGE (\d+)\]\s*---$"
    )

    HEADER_REGEX = re.compile(
        r"^(#{1,6})\s+(.*)"
    )

    # Markdown separator:
    #
    # |---|---|
    # |:---|---:|
    # ---|---|
    #
    TABLE_SEPARATOR_RE = re.compile(
        r"""
        ^\s*
        \|?
        \s*
        :?-{3,}:?
        \s*
        (?:
            \|
            \s*
            :?-{3,}:?
            \s*
        )+
        \|?
        \s*$
        """,
        re.VERBOSE,
    )

    YEAR_RE = re.compile(
        r"\b(?:19|20)\d{2}\b"
    )

    DATE_RE = re.compile(
        r"""
        \b
        (?:
            January|February|March|April|May|June|
            July|August|September|October|November|December
        )
        \s+
        \d{1,2}
        (?:
            ,\s*
            |
            \s+
        )
        (?:19|20)\d{2}
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    FINANCIAL_NUMBER_RE = re.compile(
        r"""
        ^\s*
        \(?
        [-+]?
        (?:
            \$\s*
        )?
        \d
        [\d,]*
        (?:\.\d+)?
        \%?
        \)?
        \s*
        $
        """,
        re.VERBOSE,
    )

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        max_embedding_tokens=240,
        chunk_overlap_tokens=40,
        tokenizer_encoding="cl100k_base",
        include_heading_in_text=True,
    ):
        if max_embedding_tokens <= 0:
            raise ValueError(
                "max_embedding_tokens must be > 0"
            )

        if chunk_overlap_tokens < 0:
            raise ValueError(
                "chunk_overlap_tokens must be >= 0"
            )

        if chunk_overlap_tokens >= max_embedding_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be less "
                "than max_embedding_tokens"
            )

        self.max_embedding_tokens = (
            max_embedding_tokens
        )

        self.chunk_overlap_tokens = (
            chunk_overlap_tokens
        )

        self.include_heading_in_text = (
            include_heading_in_text
        )

        try:
            self.encoder = tiktoken.get_encoding(
                tokenizer_encoding
            )
        except Exception as exc:
            raise ValueError(
                f"Unable to load tokenizer "
                f"'{tokenizer_encoding}'"
            ) from exc

        # Character splitter is only a semantic first pass.
        # Final token safety is always checked separately.
        approximate_chars = max(
            100,
            max_embedding_tokens * 3,
        )

        approximate_overlap = min(
            chunk_overlap_tokens * 3,
            approximate_chars // 4,
        )

        self.prose_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=approximate_chars,
                chunk_overlap=approximate_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    "; ",
                    ", ",
                    " ",
                    "",
                ],
                add_start_index=True,
            )
        )

    # ========================================================
    # TOKEN UTILITIES
    # ========================================================

    def count_tokens(
        self,
        text: str,
    ) -> int:
        return len(
            self.encoder.encode(
                text,
                disallowed_special=(),
            )
        )

    def fits_token_budget(
        self,
        text: str,
    ) -> bool:
        return (
            self.count_tokens(text)
            <= self.max_embedding_tokens
        )

    # ========================================================
    # HEADING UTILITIES
    # ========================================================

    @staticmethod
    def update_heading_hierarchy(
        heading_stack: Dict[int, Optional[str]],
        level: int,
        heading: str,
    ) -> None:
        heading_stack[level] = heading

        for lower_level in range(
            level + 1,
            7,
        ):
            heading_stack[lower_level] = None

    @staticmethod
    def get_heading_path(
        heading_stack: Dict[int, Optional[str]],
    ) -> List[str]:
        return [
            heading
            for _, heading in sorted(
                heading_stack.items()
            )
            if heading
        ]

    @staticmethod
    def clean_heading(
        heading: str,
    ) -> str:
        heading = heading.strip()

        if (
            heading.startswith("**")
            and heading.endswith("**")
        ):
            heading = heading[2:-2].strip()

        if (
            heading.startswith("__")
            and heading.endswith("__")
        ):
            heading = heading[2:-2].strip()

        return heading

    def _heading_context(
        self,
        heading_path: List[str],
    ) -> str:
        if not heading_path:
            return ""

        return (
            "Section: "
            + " > ".join(heading_path)
        )

    def _format_text_with_heading(
        self,
        text: str,
        heading_path: List[str],
    ) -> str:
        text = text.strip()

        if not text:
            return ""

        if (
            self.include_heading_in_text
            and heading_path
        ):
            return (
                self._heading_context(
                    heading_path
                )
                + "\n\n"
                + text
            )

        return text

    # ========================================================
    # TABLE DETECTION
    # ========================================================

    @classmethod
    def is_table_row(
        cls,
        line: str,
    ) -> bool:
        stripped = line.strip()

        return (
            bool(stripped)
            and "|" in stripped
            and not stripped.startswith("```")
        )

    @classmethod
    def is_table_separator(
        cls,
        line: str,
    ) -> bool:
        return bool(
            cls.TABLE_SEPARATOR_RE.match(
                line
            )
        )

    @classmethod
    def is_table_start(
        cls,
        lines: List[str],
        index: int,
    ) -> bool:
        if index + 1 >= len(lines):
            return False

        return (
            cls.is_table_row(
                lines[index]
            )
            and cls.is_table_separator(
                lines[index + 1]
            )
        )

    @staticmethod
    def parse_table_cells(
        line: str,
    ) -> List[str]:
        stripped = line.strip()

        if stripped.startswith("|"):
            stripped = stripped[1:]

        if stripped.endswith("|"):
            stripped = stripped[:-1]

        return [
            cell.strip()
            for cell in stripped.split("|")
        ]

    @classmethod
    def is_financial_number(
        cls,
        value: str,
    ) -> bool:
        value = value.strip()

        if not value:
            return False

        if value in {
            "-",
            "—",
            "–",
            "N/A",
            "n/a",
        }:
            return True

        return bool(
            cls.FINANCIAL_NUMBER_RE.fullmatch(
                value
            )
        )

    @classmethod
    def _looks_like_header_row(
        cls,
        cells: List[str],
        expected_columns: Optional[int] = None,
    ) -> bool:
        """
        Classify a candidate row occurring immediately after
        the Markdown separator.
        """

        if not cells:
            return False

        populated = [
            cell.strip()
            for cell in cells
            if cell.strip()
        ]

        if not populated:
            return False

        if (
            expected_columns is not None
            and len(cells) != expected_columns
        ):
            return False

        # ----------------------------------------------------
        # Strong date/year header.
        # ----------------------------------------------------

        year_count = sum(
            bool(
                cls.YEAR_RE.search(cell)
            )
            for cell in populated
        )

        date_count = sum(
            bool(
                cls.DATE_RE.search(cell)
            )
            for cell in populated
        )

        if year_count >= 2:
            return True

        if date_count >= 2:
            return True

        # ----------------------------------------------------
        # Markdown multi-line header cells.
        # ----------------------------------------------------

        if any(
            "<br>" in cell.lower()
            for cell in populated
        ):
            return True

        # ----------------------------------------------------
        # Strong bold-header pattern.
        # ----------------------------------------------------

        bold_count = sum(
            "**" in cell
            for cell in populated
        )

        if (
            len(populated) >= 2
            and bold_count >= 2
            and bold_count >= len(populated) * 0.5
        ):
            return True

        # ----------------------------------------------------
        # Textual column labels.
        # ----------------------------------------------------

        numeric_count = sum(
            cls.is_financial_number(cell)
            for cell in populated
        )

        if (
            len(populated) >= 2
            and numeric_count == 0
        ):
            return True

        return False

    # ========================================================
    # PARSING
    # ========================================================

    def _parse_blocks(
        self,
        markdown: str,
    ) -> List[ContentBlock]:

        lines = markdown.splitlines()

        blocks: List[
            ContentBlock
        ] = []

        current_page: Optional[int] = None

        heading_stack: Dict[
            int,
            Optional[str],
        ] = {
            level: None
            for level in range(1, 7)
        }

        paragraph_segments: List[
            PageSegment
        ] = []

        current_paragraph_lines: List[
            str
        ] = []

        current_paragraph_page: Optional[
            int
        ] = None

        def flush_paragraph() -> None:
            nonlocal current_paragraph_lines
            nonlocal current_paragraph_page

            if (
                current_paragraph_lines
                and current_paragraph_page is not None
            ):
                text = "\n".join(
                    current_paragraph_lines
                ).strip()

                if text:
                    paragraph_segments.append(
                        PageSegment(
                            text=text,
                            page=current_paragraph_page,
                        )
                    )

            current_paragraph_lines = []
            current_paragraph_page = None

        def flush_paragraph_block() -> None:
            nonlocal paragraph_segments

            flush_paragraph()

            if paragraph_segments:
                blocks.append(
                    ContentBlock(
                        block_type="paragraph",
                        segments=list(
                            paragraph_segments
                        ),
                        heading_path=(
                            self.get_heading_path(
                                heading_stack
                            )
                        ),
                    )
                )

                paragraph_segments = []

        def add_paragraph_line(
            text: str,
            page: int,
        ) -> None:
            nonlocal current_paragraph_page

            if (
                current_paragraph_page is not None
                and current_paragraph_page != page
            ):
                flush_paragraph()

            current_paragraph_page = page
            current_paragraph_lines.append(
                text
            )

        i = 0

        while i < len(lines):

            raw_line = lines[i]
            stripped = raw_line.strip()

            # =================================================
            # PAGE START
            # =================================================

            page_match = (
                self.PAGE_START_REGEX.match(
                    stripped
                )
            )

            if page_match:
                current_page = int(
                    page_match.group(1)
                )

                i += 1
                continue

            # =================================================
            # PAGE END
            # =================================================

            if self.PAGE_END_REGEX.match(
                stripped
            ):
                i += 1
                continue

            # =================================================
            # EMPTY
            # =================================================

            if not stripped:
                flush_paragraph()
                i += 1
                continue

            # =================================================
            # HEADING
            # =================================================

            heading_match = (
                self.HEADER_REGEX.match(
                    stripped
                )
            )

            if heading_match:

                flush_paragraph_block()

                level = len(
                    heading_match.group(1)
                )

                heading = self.clean_heading(
                    heading_match.group(2)
                )

                self.update_heading_hierarchy(
                    heading_stack,
                    level,
                    heading,
                )

                i += 1
                continue

            # =================================================
            # TABLE
            # =================================================

            if self.is_table_start(
                lines,
                i,
            ):

                flush_paragraph_block()

                table_lines: List[
                    str
                ] = []

                table_pages: List[
                    int
                ] = []

                # ------------------------------------------------
                # Read complete table, including page transitions.
                # ------------------------------------------------

                while i < len(lines):

                    table_line = lines[i]

                    table_stripped = (
                        table_line.strip()
                    )

                    page_match = (
                        self.PAGE_START_REGEX.match(
                            table_stripped
                        )
                    )

                    if page_match:
                        current_page = int(
                            page_match.group(1)
                        )

                        i += 1
                        continue

                    if self.PAGE_END_REGEX.match(
                        table_stripped
                    ):
                        i += 1
                        continue

                    if self.is_table_row(
                        table_line
                    ):

                        table_lines.append(
                            table_stripped
                        )

                        table_pages.append(
                            current_page
                            if current_page is not None
                            else 0
                        )

                        i += 1
                        continue

                    break

                # ------------------------------------------------
                # Need at least a header + separator.
                # ------------------------------------------------

                if len(table_lines) >= 2:

                    separator_idx: Optional[
                        int
                    ] = None

                    for idx, line in enumerate(
                        table_lines
                    ):

                        if self.is_table_separator(
                            line
                        ):
                            separator_idx = idx
                            break

                    if separator_idx is None:

                        logger.warning(
                            "Table detected but no "
                            "separator row found."
                        )

                        for line, page in zip(
                            table_lines,
                            table_pages,
                        ):
                            add_paragraph_line(
                                line,
                                page,
                            )

                        continue

                    # ------------------------------------------------
                    # Determine column count from separator.
                    # ------------------------------------------------

                    separator_cells = (
                        self.parse_table_cells(
                            table_lines[
                                separator_idx
                            ]
                        )
                    )

                    expected_columns = len(
                        separator_cells
                    )

                    # ------------------------------------------------
                    # Header = everything through separator,
                    # plus valid continuation rows immediately after.
                    # ------------------------------------------------

                    header_end_idx = (
                        separator_idx
                    )

                    for idx in range(
                        separator_idx + 1,
                        len(table_lines),
                    ):

                        candidate = (
                            table_lines[idx]
                        )

                        if self.is_table_separator(
                            candidate
                        ):
                            break

                        cells = (
                            self.parse_table_cells(
                                candidate
                            )
                        )

                        if not self._looks_like_header_row(
                            cells,
                            expected_columns,
                        ):
                            break

                        header_end_idx = idx

                    table_header = "\n".join(
                        table_lines[
                            : header_end_idx + 1
                        ]
                    )

                    table_rows = table_lines[
                        header_end_idx + 1:
                    ]

                    table_row_pages = (
                        table_pages[
                            header_end_idx + 1:
                        ]
                    )

                    header_pages = (
                        table_pages[
                            : header_end_idx + 1
                        ]
                    )

                    blocks.append(
                        ContentBlock(
                            block_type="table",

                            segments=[
                                PageSegment(
                                    text="",
                                    page=page,
                                )
                                for page in sorted(
                                    set(
                                        header_pages
                                        + table_row_pages
                                    )
                                )
                            ],

                            heading_path=(
                                self.get_heading_path(
                                    heading_stack
                                )
                            ),

                            table_header=(
                                table_header
                            ),

                            table_rows=list(
                                table_rows
                            ),

                            table_row_pages=list(
                                table_row_pages
                            ),
                        )
                    )

                continue

            # =================================================
            # NORMAL CONTENT
            # =================================================

            if current_page is None:

                logger.warning(
                    "Content encountered before "
                    "page marker: %s",
                    stripped[:100],
                )

                i += 1
                continue

            add_paragraph_line(
                raw_line,
                current_page,
            )

            i += 1

        # =====================================================
        # FINAL PARAGRAPH
        # =====================================================

        flush_paragraph_block()

        return blocks

    # ========================================================
    # SPATIAL CHARACTER MAPPING
    # ========================================================

    def _build_continuous_text_and_map(
        self,
        segments: List[PageSegment],
    ) -> Tuple[
        str,
        List[Tuple[int, int, int]],
    ]:

        parts: List[str] = []

        page_intervals: List[
            Tuple[int, int, int]
        ] = []

        current_offset = 0

        for segment in segments:

            if not segment.text:
                continue

            if parts:
                parts.append("\n")
                current_offset += 1

            start = current_offset

            parts.append(
                segment.text
            )

            current_offset += len(
                segment.text
            )

            end = current_offset

            page_intervals.append(
                (
                    start,
                    end,
                    segment.page,
                )
            )

        return (
            "".join(parts),
            page_intervals,
        )

    def _intersect_page_ranges(
        self,
        chunk_start: int,
        chunk_end: int,
        page_intervals: List[
            Tuple[int, int, int]
        ],
    ) -> List[
        Tuple[int, int, int]
    ]:

        spans = []

        for (
            page_start,
            page_end,
            page,
        ) in page_intervals:

            start = max(
                chunk_start,
                page_start,
            )

            end = min(
                chunk_end,
                page_end,
            )

            if start < end:
                spans.append(
                    (
                        start,
                        end,
                        page,
                    )
                )

        return spans

    def _relative_page_spans(
        self,
        page_spans: List[
            Tuple[int, int, int]
        ],
        chunk_start: int,
    ) -> List[Dict]:

        return [
            {
                "page": page,
                "start": start - chunk_start,
                "end": end - chunk_start,
            }
            for (
                start,
                end,
                page,
            ) in page_spans
        ]

    # ========================================================
    # PROSE SPLITTING
    # ========================================================

    def _split_paragraph(
        self,
        block: ContentBlock,
    ) -> List[ContentBlock]:

        (
            continuous_text,
            page_intervals,
        ) = self._build_continuous_text_and_map(
            block.segments
        )

        if not continuous_text.strip():
            return []

        full_text = (
            self._format_text_with_heading(
                continuous_text,
                block.heading_path,
            )
        )

        if self.fits_token_budget(
            full_text
        ):
            return [block]

        heading_context = ""

        if (
            self.include_heading_in_text
            and block.heading_path
        ):
            heading_context = (
                self._heading_context(
                    block.heading_path
                )
            )

        heading_tokens = (
            self.count_tokens(
                heading_context
            )
            if heading_context
            else 0
        )

        available_tokens = (
            self.max_embedding_tokens
            - heading_tokens
        )

        if available_tokens <= 0:
            raise ValueError(
                "Heading hierarchy alone exceeds "
                "the configured embedding token budget."
            )

        approximate_chars = max(
            100,
            available_tokens * 3,
        )

        approximate_overlap = min(
            self.chunk_overlap_tokens * 3,
            approximate_chars // 4,
        )

        splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=approximate_chars,
                chunk_overlap=approximate_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    "; ",
                    ", ",
                    " ",
                    "",
                ],
                add_start_index=True,
            )
        )

        docs = splitter.create_documents(
            [continuous_text]
        )

        results: List[
            ContentBlock
        ] = []

        for doc in docs:

            chunk_text = (
                doc.page_content
            )

            chunk_start = int(
                doc.metadata[
                    "start_index"
                ]
            )

            chunk_end = (
                chunk_start
                + len(chunk_text)
            )

            page_spans = (
                self._intersect_page_ranges(
                    chunk_start,
                    chunk_end,
                    page_intervals,
                )
            )

            if not page_spans:

                page_spans = [
                    (
                        chunk_start,
                        chunk_end,
                        page,
                    )
                    for page in sorted(
                        block.pages
                    )
                ]

            candidate_text = (
                self._format_text_with_heading(
                    chunk_text,
                    block.heading_path,
                )
            )

            if self.fits_token_budget(
                candidate_text
            ):

                results.append(
                    ContentBlock(
                        block_type="paragraph",

                        segments=[
                            PageSegment(
                                text=chunk_text,
                                page=(
                                    page_spans[0][2]
                                ),
                            )
                        ],

                        heading_path=list(
                            block.heading_path
                        ),

                        page_spans=(
                            self._relative_page_spans(
                                page_spans,
                                chunk_start,
                            )
                        ),
                    )
                )

                continue

            # ------------------------------------------------
            # Deterministic token-safe fallback.
            # ------------------------------------------------

            safe_ranges = (
                self._token_safe_ranges(
                    text=chunk_text,
                    original_start=chunk_start,
                    heading_path=(
                        block.heading_path
                    ),
                )
            )

            for (
                safe_text,
                safe_start,
                safe_end,
            ) in safe_ranges:

                safe_page_spans = (
                    self._intersect_page_ranges(
                        safe_start,
                        safe_end,
                        page_intervals,
                    )
                )

                if not safe_page_spans:

                    safe_page_spans = [
                        (
                            safe_start,
                            safe_end,
                            page,
                        )
                        for page in sorted(
                            block.pages
                        )
                    ]

                results.append(
                    ContentBlock(
                        block_type="paragraph",

                        segments=[
                            PageSegment(
                                text=safe_text,
                                page=(
                                    safe_page_spans[0][2]
                                ),
                            )
                        ],

                        heading_path=list(
                            block.heading_path
                        ),

                        page_spans=(
                            self._relative_page_spans(
                                safe_page_spans,
                                safe_start,
                            )
                        ),
                    )
                )

        return results

    # ========================================================
    # TOKEN-SAFE PROSE FALLBACK
    # ========================================================

    def _token_safe_ranges(
        self,
        text: str,
        original_start: int,
        heading_path: List[str],
    ) -> List[
        Tuple[str, int, int]
    ]:

        heading_context = ""

        if (
            self.include_heading_in_text
            and heading_path
        ):
            heading_context = (
                self._heading_context(
                    heading_path
                )
            )

        heading_tokens = (
            self.count_tokens(
                heading_context
            )
            if heading_context
            else 0
        )

        available_tokens = (
            self.max_embedding_tokens
            - heading_tokens
        )

        if available_tokens <= 0:
            raise ValueError(
                "Heading context exceeds "
                "embedding token budget."
            )

        encoded = self.encoder.encode(
            text,
            disallowed_special=(),
        )

        if not encoded:
            return []

        results: List[
            Tuple[str, int, int]
        ] = []

        token_start = 0
        search_offset = 0

        while token_start < len(encoded):

            token_end = min(
                token_start
                + available_tokens,
                len(encoded),
            )

            selected_text: Optional[
                str
            ] = None

            selected_token_count = 0

            while token_end > token_start:

                candidate_text = (
                    self.encoder.decode(
                        encoded[
                            token_start:token_end
                        ]
                    )
                )

                if not candidate_text:
                    token_end -= 1
                    continue

                formatted = (
                    self._format_text_with_heading(
                        candidate_text,
                        heading_path,
                    )
                )

                if self.fits_token_budget(
                    formatted
                ):
                    selected_text = (
                        candidate_text
                    )

                    selected_token_count = (
                        token_end
                        - token_start
                    )

                    break

                token_end -= 1

            if selected_text is None:
                raise RuntimeError(
                    "Unable to create token-safe "
                    "prose chunk."
                )

            relative_start = text.find(
                selected_text,
                search_offset,
            )

            if relative_start < 0:
                raise RuntimeError(
                    "Unable to map token-safe "
                    "piece to original text."
                )

            relative_end = (
                relative_start
                + len(selected_text)
            )

            absolute_start = (
                original_start
                + relative_start
            )

            absolute_end = (
                original_start
                + relative_end
            )

            results.append(
                (
                    selected_text,
                    absolute_start,
                    absolute_end,
                )
            )

            search_offset = relative_end

            if selected_token_count <= 0:
                raise RuntimeError(
                    "Token-safe prose splitting "
                    "made no token progress."
                )

            token_start += (
                selected_token_count
            )

        return results

    # ========================================================
    # TABLE UTILITIES
    # ========================================================

    @staticmethod
    def _build_table(
        header: Optional[str],
        rows: List[str],
    ) -> str:

        parts: List[str] = []

        if header:
            parts.append(header)

        parts.extend(rows)

        return "\n".join(parts)

    def _format_table_text(
        self,
        block: ContentBlock,
        rows: List[str],
    ) -> str:

        table = self._build_table(
            block.table_header,
            rows,
        )

        return (
            self._format_text_with_heading(
                table,
                block.heading_path,
            )
        )

    # ========================================================
    # TABLE ROW TOKEN UTILITIES
    # ========================================================

    def _table_row_prefix(
        self,
        block: ContentBlock,
    ) -> str:
        """
        Return everything that must accompany an individual
        table-row piece.

        Prefix:

            heading
            +
            complete table header
        """

        table_header = (
            block.table_header
            or ""
        ).strip()

        if not table_header:
            raise ValueError(
                "Cannot split a table row without "
                "a table header."
            )

        if (
            self.include_heading_in_text
            and block.heading_path
        ):
            return (
                self._heading_context(
                    block.heading_path
                )
                + "\n\n"
                + table_header
            )

        return table_header

    def _table_row_piece_fits(
        self,
        block: ContentBlock,
        row_piece: str,
    ) -> bool:
        """
        Check:

            heading + table header + row piece

        against the actual configured token budget.
        """

        candidate = (
            self._format_table_text(
                block,
                [row_piece],
            )
        )

        return self.fits_token_budget(
            candidate
        )

    def _token_safe_table_row_ranges(
        self,
        block: ContentBlock,
        row: str,
    ) -> List[str]:
        """
        Split an oversized table row into token-safe pieces.

        Every piece is validated against:

            heading
            +
            complete table header
            +
            row piece
        """

        prefix = self._table_row_prefix(
            block
        )

        prefix_tokens = self.count_tokens(
            prefix
        )

        available_tokens = (
            self.max_embedding_tokens
            - prefix_tokens
        )

        if available_tokens <= 0:
            raise ValueError(
                "Table heading and header alone "
                "exceed the configured embedding "
                "token budget."
            )

        encoded = self.encoder.encode(
            row,
            disallowed_special=(),
        )

        if not encoded:
            return []

        pieces: List[str] = []

        token_start = 0

        while token_start < len(encoded):

            token_end = min(
                token_start
                + available_tokens,
                len(encoded),
            )

            selected_piece: Optional[
                str
            ] = None

            while token_end > token_start:

                candidate_piece = (
                    self.encoder.decode(
                        encoded[
                            token_start:token_end
                        ]
                    )
                )

                if not candidate_piece:
                    token_end -= 1
                    continue

                if self._table_row_piece_fits(
                    block,
                    candidate_piece,
                ):
                    selected_piece = (
                        candidate_piece
                    )
                    break

                token_end -= 1

            if selected_piece is None:
                raise RuntimeError(
                    "Unable to create a token-safe "
                    "table-row piece while preserving "
                    "the complete table header."
                )

            pieces.append(
                selected_piece
            )

            consumed_tokens = len(
                self.encoder.encode(
                    selected_piece,
                    disallowed_special=(),
                )
            )

            if consumed_tokens <= 0:
                raise RuntimeError(
                    "Token-safe table-row splitting "
                    "made no token progress."
                )

            token_start += (
                consumed_tokens
            )

        return pieces

    # ========================================================
    # TABLE SPLITTING
    # ========================================================

    def _split_table(
        self,
        block: ContentBlock,
    ) -> List[ContentBlock]:
        """
        Split oversized tables only at row boundaries.

        Every normal table chunk contains:

            heading
            complete multi-line table header
            one or more complete data rows
        """

        if not block.table_header:
            return [block]

        if not block.table_rows:

            header_text = (
                self._format_table_text(
                    block,
                    [],
                )
            )

            if self.fits_token_budget(
                header_text
            ):
                return [block]

            raise ValueError(
                "Table header exceeds the configured "
                "embedding token budget. "
                "Cannot split a table header without "
                "breaking table structure."
            )

        # ----------------------------------------------------
        # Validate source provenance.
        # ----------------------------------------------------

        if len(block.table_rows) != len(
            block.table_row_pages
        ):
            raise ValueError(
                "table_rows and table_row_pages must "
                "have identical lengths before table splitting."
            )

        full_table_text = (
            self._format_table_text(
                block,
                block.table_rows,
            )
        )

        if self.fits_token_budget(
            full_table_text
        ):
            return [block]

        row_pages = list(
            block.table_row_pages
        )

        results: List[
            ContentBlock
        ] = []

        current_rows: List[str] = []
        current_row_pages: List[int] = []

        for index, row in enumerate(
            block.table_rows
        ):

            row_page = row_pages[index]

            candidate_rows = (
                current_rows + [row]
            )

            candidate_text = (
                self._format_table_text(
                    block,
                    candidate_rows,
                )
            )

            # ------------------------------------------------
            # Candidate fits.
            # ------------------------------------------------

            if self.fits_token_budget(
                candidate_text
            ):

                current_rows.append(
                    row
                )

                current_row_pages.append(
                    row_page
                )

                continue

            # ------------------------------------------------
            # Flush existing rows.
            # ------------------------------------------------

            if current_rows:

                results.append(
                    self._make_table_block(
                        block=block,
                        rows=current_rows,
                        row_pages=current_row_pages,
                    )
                )

                current_rows = []
                current_row_pages = []

            # ------------------------------------------------
            # Try individual row.
            # ------------------------------------------------

            single_row_text = (
                self._format_table_text(
                    block,
                    [row],
                )
            )

            if self.fits_token_budget(
                single_row_text
            ):

                current_rows = [
                    row
                ]

                current_row_pages = [
                    row_page
                ]

                continue

            # ------------------------------------------------
            # Individual row too large.
            # ------------------------------------------------

            logger.warning(
                "Table row exceeds token budget. "
                "Page=%s",
                row_page,
            )

            oversized_chunks = (
                self._split_oversized_table_row(
                    block=block,
                    row=row,
                    page=row_page,
                )
            )

            results.extend(
                oversized_chunks
            )

        # ----------------------------------------------------
        # Flush final rows.
        # ----------------------------------------------------

        if current_rows:

            results.append(
                self._make_table_block(
                    block=block,
                    rows=current_rows,
                    row_pages=current_row_pages,
                )
            )

        # ----------------------------------------------------
        # Defensive validation.
        # ----------------------------------------------------

        for result in results:

            if len(
                result.table_rows
            ) != len(
                result.table_row_pages
            ):
                raise RuntimeError(
                    "Table chunk has mismatched "
                    "row/page provenance."
                )

            formatted = (
                self._format_table_text(
                    result,
                    result.table_rows,
                )
            )

            if not self.fits_token_budget(
                formatted
            ):
                raise RuntimeError(
                    "Table splitting produced a chunk "
                    "that exceeds the token budget."
                )

        return results

    def _make_table_block(
        self,
        block: ContentBlock,
        rows: List[str],
        row_pages: List[int],
    ) -> ContentBlock:
        """
        Construct a table block while preserving:

            rows[i] <-> row_pages[i]
        """

        if len(rows) != len(
            row_pages
        ):
            raise ValueError(
                "rows and row_pages must have "
                "identical lengths."
            )

        return ContentBlock(
            block_type="table",

            segments=[
                PageSegment(
                    text="",
                    page=page,
                )
                for page in sorted(
                    set(row_pages)
                )
            ],

            heading_path=list(
                block.heading_path
            ),

            table_header=(
                block.table_header
            ),

            table_rows=list(
                rows
            ),

            table_row_pages=list(
                row_pages
            ),
        )

    def _split_oversized_table_row(
        self,
        block: ContentBlock,
        row: str,
        page: int,
    ) -> List[ContentBlock]:
        """
        Last-resort handling for an individually oversized row.

        The complete table header is preserved in every
        resulting chunk.

        Every emitted chunk is:

            heading
            complete table header
            row piece

        and is guaranteed to fit the token budget.
        """

        if not block.table_header:
            raise ValueError(
                "Cannot split an oversized table row "
                "without a table header."
            )

        # ----------------------------------------------------
        # First verify whether complete row fits.
        # ----------------------------------------------------

        full_text = (
            self._format_table_text(
                block,
                [row],
            )
        )

        if self.fits_token_budget(
            full_text
        ):

            return [
                self._make_table_block(
                    block=block,
                    rows=[row],
                    row_pages=[page],
                )
            ]

        logger.warning(
            "Splitting oversized table row "
            "on page %s.",
            page,
        )

        pieces = (
            self._token_safe_table_row_ranges(
                block=block,
                row=row,
            )
        )

        results: List[
            ContentBlock
        ] = []

        for piece in pieces:

            candidate = (
                self._make_table_block(
                    block=block,
                    rows=[piece],
                    row_pages=[page],
                )
            )

            formatted = (
                self._format_table_text(
                    candidate,
                    [piece],
                )
            )

            token_count = (
                self.count_tokens(
                    formatted
                )
            )

            if token_count > (
                self.max_embedding_tokens
            ):
                raise RuntimeError(
                    "Oversized table-row fallback "
                    "still exceeds token budget: "
                    f"{token_count} > "
                    f"{self.max_embedding_tokens}"
                )

            results.append(
                candidate
            )

        return results

    # ========================================================
    # FINAL CHUNK CREATION
    # ========================================================

    def _create_final_chunk(
        self,
        block: ContentBlock,
        source: str,
        chunk_id: int,
        table_chunk_index: Optional[int] = None,
        table_chunk_count: Optional[int] = None,
    ) -> Optional[Dict]:

        if block.block_type == "table":

            content = self._build_table(
                block.table_header,
                block.table_rows,
            )

        else:

            content = block.text

        if not content.strip():
            return None

        text = (
            self._format_text_with_heading(
                content,
                block.heading_path,
            )
        )

        token_count = (
            self.count_tokens(
                text
            )
        )

        if token_count > (
            self.max_embedding_tokens
        ):
            raise RuntimeError(
                f"Chunk {chunk_id} exceeds "
                f"token budget: "
                f"{token_count} > "
                f"{self.max_embedding_tokens}"
            )

        # ----------------------------------------------------
        # Page provenance.
        # ----------------------------------------------------

        pages = sorted(
            block.pages
        )

        page_start = (
            pages[0]
            if pages
            else None
        )

        page_end = (
            pages[-1]
            if pages
            else None
        )

        heading_path = list(
            block.heading_path
        )

        metadata = {
            "source": os.path.basename(
                source
            ),

            "chunk_id": (
                f"{os.path.basename(source)}"
                f"_{chunk_id:05d}"
            ),

            "pages": pages,

            "page_start": page_start,

            "page_end": page_end,

            "page_spans": (
                block.page_spans
                if block.page_spans
                else []
            ),

            "heading_path": heading_path,

            "heading": (
                heading_path[-1]
                if heading_path
                else None
            ),

            "content_type": (
                block.block_type
            ),

            "token_count": token_count,
        }

        for index, heading in enumerate(
            heading_path,
            start=1,
        ):
            metadata[
                f"header_{index}"
            ] = heading

        if table_chunk_index is not None:
            metadata[
                "table_chunk_index"
            ] = table_chunk_index

        if table_chunk_count is not None:
            metadata[
                "table_chunk_count"
            ] = table_chunk_count

        return {
            "text": text,
            "metadata": metadata,
        }

    # ========================================================
    # PUBLIC API
    # ========================================================

    def chunk_document(
        self,
        full_markdown: str,
        source: str = "unknown",
    ) -> List[Dict]:

        if not full_markdown.strip():

            logger.warning(
                "Markdown document is empty."
            )

            return []

        blocks = self._parse_blocks(
            full_markdown
        )

        logger.info(
            "Parsed %d semantic blocks from %s",
            len(blocks),
            source,
        )

        final_chunks: List[
            Dict
        ] = []

        chunk_id = 0

        for block in blocks:

            # =================================================
            # TABLE
            # =================================================

            if block.block_type == "table":

                table_chunks = (
                    self._split_table(
                        block
                    )
                )

                table_count = len(
                    table_chunks
                )

                for (
                    table_index,
                    table_block,
                ) in enumerate(
                    table_chunks,
                    start=1,
                ):

                    result = (
                        self._create_final_chunk(
                            block=table_block,
                            source=source,
                            chunk_id=chunk_id,
                            table_chunk_index=(
                                table_index
                            ),
                            table_chunk_count=(
                                table_count
                            ),
                        )
                    )

                    if result:

                        final_chunks.append(
                            result
                        )

                        chunk_id += 1

                continue

            # =================================================
            # PARAGRAPH
            # =================================================

            paragraph_chunks = (
                self._split_paragraph(
                    block
                )
            )

            for paragraph_block in (
                paragraph_chunks
            ):

                result = (
                    self._create_final_chunk(
                        block=paragraph_block,
                        source=source,
                        chunk_id=chunk_id,
                    )
                )

                if result:

                    final_chunks.append(
                        result
                    )

                    chunk_id += 1

        logger.info(
            "Created %d chunks from %s",
            len(final_chunks),
            source,
        )

        return final_chunks

    # ========================================================
    # LANGCHAIN / CHROMA INTEGRATION
    # ========================================================

    def chunk_to_documents(
        self,
        full_markdown: str,
        source: str = "unknown",
    ) -> List[Document]:

        chunks = self.chunk_document(
            full_markdown=full_markdown,
            source=source,
        )

        return [
            Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"],
            )
            for chunk in chunks
        ]


# ============================================================
# CLI / TEST
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = Path(
        settings.PROJECT_ROOT
    )

    sample_md_path = (
        PROJECT_ROOT
        / settings.MARKDOWN_DIR
        / "sample.md"
    )

    if not sample_md_path.is_file():

        raise FileNotFoundError(
            f"Markdown file not found: "
            f"{sample_md_path}"
        )

    with open(
        sample_md_path,
        "r",
        encoding="utf-8",
    ) as file:

        markdown = file.read()

    chunker = FinTechMarkdownChunker(
        max_embedding_tokens=(
            settings.CHUNK_SIZE
        ),

        chunk_overlap_tokens=(
            settings.CHUNK_OVERLAP
        ),

        tokenizer_encoding=(
            settings.TOKENIZER_ENCODING
        ),

        include_heading_in_text=True,
    )

    chunks = chunker.chunk_document(
        markdown,
        source=str(sample_md_path),
    )

    print(
        f"\n{'=' * 80}"
    )

    print(
        f"TOTAL CHUNKS: {len(chunks)}"
    )

    print(
        f"{'=' * 80}"
    )

    for index, chunk in enumerate(
        chunks[:20],
        start=1,
    ):

        metadata = chunk[
            "metadata"
        ]

        print(
            f"\n[CHUNK {index}]"
        )

        print(
            f"ID: "
            f"{metadata['chunk_id']}"
        )

        print(
            f"Type: "
            f"{metadata['content_type']}"
        )

        print(
            f"Pages: "
            f"{metadata['pages']}"
        )

        print(
            f"Page Start: "
            f"{metadata['page_start']}"
        )

        print(
            f"Page End: "
            f"{metadata['page_end']}"
        )

        print(
            f"Heading Path: "
            f"{metadata['heading_path']}"
        )

        print(
            f"Token Count: "
            f"{metadata['token_count']}"
        )

        if "table_chunk_index" in metadata:

            print(
                f"Table Chunk: "
                f"{metadata['table_chunk_index']}/"
                f"{metadata['table_chunk_count']}"
            )

        print(
            f"Page Spans: "
            f"{metadata['page_spans']}"
        )

        print("\nTEXT:")

        print(
            chunk["text"]
        )

        print(
            "-" * 80
        )
