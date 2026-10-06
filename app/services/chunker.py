import logging
import os
import re
from typing import Dict, List, Optional
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


logger = logging.getLogger(__name__)


class FinTechMarkdownChunker:
    """
    Markdown-aware and page-aware chunker for financial documents.

    Features:
    - Understands Markdown headings (#, ##, ###).
    - Allows sections to span multiple PDF pages.
    - Preserves heading hierarchy.
    - Tracks the exact pages contributing to each chunk.
    - Stores only page_start and page_end metadata.
    - Uses RecursiveCharacterTextSplitter for oversized sections.
    """

    PAGE_START_REGEX = re.compile(
        r"^---\s*\[START OF PAGE (\d+)\]\s*---$"
    )

    PAGE_END_REGEX = re.compile(
        r"^---\s*\[END OF PAGE (\d+)\]\s*---$"
    )

    HEADER_REGEX = re.compile(
        r"^(#{1,3})\s+(.*)"
    )

    # Internal marker used while processing chunks.
    INTERNAL_PAGE_REGEX = re.compile(
        r"<!--\s*PAGE:(\d+)\s*-->"
    )

    def __init__(
        self,
        max_chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        if max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size must be greater than 0"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap must be >= 0"
            )

        if chunk_overlap >= max_chunk_size:
            raise ValueError(
                "chunk_overlap must be less than max_chunk_size"
            )

        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
            add_start_index=True,
        )

    # ------------------------------------------------------------------
    # PAGE METADATA
    # ------------------------------------------------------------------

    @staticmethod
    def get_page_range(
        pages: List[int],
    ) -> Dict[str, Optional[int]]:
        """
        Convert a list of page numbers into:

            {
                "page_start": 20,
                "page_end": 21
            }

        We intentionally do not store:
            page = "20-21"
            pages = [20, 21]
        """

        if not pages:
            return {
                "page_start": None,
                "page_end": None,
            }

        unique_pages = sorted(set(pages))

        return {
            "page_start": unique_pages[0],
            "page_end": unique_pages[-1],
        }

    # ------------------------------------------------------------------
    # HEADING HIERARCHY
    # ------------------------------------------------------------------

    @staticmethod
    def update_heading_hierarchy(
        heading_stack: Dict[int, Optional[str]],
        level: int,
        heading: str,
    ) -> None:
        """
        Maintain Markdown heading hierarchy.

        Example:

            # Financial Performance
            ## Revenue
            ### Domestic Revenue

        becomes:

            header_1 = "Financial Performance"
            header_2 = "Revenue"
            header_3 = "Domestic Revenue"

        When a new ## heading appears, the previous
        ### heading is cleared.
        """

        heading_stack[level] = heading

        for lower_level in range(
            level + 1,
            4,
        ):
            heading_stack[lower_level] = None

    # ------------------------------------------------------------------
    # PARSE MARKDOWN INTO LOGICAL SECTIONS
    # ------------------------------------------------------------------

    def _parse_sections(
        self,
        full_markdown: str,
    ) -> List[Dict]:
        """
        Parse page-marked Markdown into logical sections.

        Important:

        We DO NOT split the document page-by-page.

        A section is allowed to continue from:

            page 20 → page 21 → page 22

        Page numbers are tracked while the section is being built.
        """

        lines = full_markdown.splitlines()

        sections: List[Dict] = []

        current_lines: List[str] = []
        current_pages = set()

        current_page: Optional[int] = None

        heading_stack: Dict[
            int,
            Optional[str],
        ] = {
            1: None,
            2: None,
            3: None,
        }

        def flush_section() -> None:
            """
            Save the current section.
            """

            nonlocal current_lines
            nonlocal current_pages

            text = "\n".join(
                current_lines
            ).strip()

            if not text:
                return

            sections.append({
                "text": text,
                "pages": set(current_pages),
                "headers": heading_stack.copy(),
            })

        for line in lines:
            stripped = line.strip()

            # ----------------------------------------------------------
            # PAGE START
            # ----------------------------------------------------------

            page_start_match = (
                self.PAGE_START_REGEX.match(
                    stripped
                )
            )

            if page_start_match:
                current_page = int(
                    page_start_match.group(1)
                )

                # Add an internal marker.
                #
                # This marker allows the fallback splitter to
                # determine exactly which pages belong to each
                # resulting chunk.
                current_lines.append(
                    f"<!-- PAGE:{current_page} -->"
                )

                continue

            # ----------------------------------------------------------
            # PAGE END
            # ----------------------------------------------------------

            if self.PAGE_END_REGEX.match(stripped):
                continue

            # ----------------------------------------------------------
            # MARKDOWN HEADER
            # ----------------------------------------------------------

            header_match = self.HEADER_REGEX.match(
                stripped
            )

            if header_match:

                # Save the previous section.
                flush_section()

                level = len(
                    header_match.group(1)
                )

                heading = header_match.group(2).strip()

                self.update_heading_hierarchy(
                    heading_stack,
                    level,
                    heading,
                )

                # Start the new section with its header.
                current_lines = [
                    stripped
                ]

                current_pages = set()

                if current_page is not None:
                    current_pages.add(
                        current_page
                    )

                continue

            # ----------------------------------------------------------
            # NORMAL TEXT
            # ----------------------------------------------------------

            current_lines.append(line)

            if stripped and current_page is not None:
                current_pages.add(
                    current_page
                )

        # Save final section.
        flush_section()

        return sections

    # ------------------------------------------------------------------
    # FIND PAGES INSIDE A CHUNK
    # ------------------------------------------------------------------

    @classmethod
    def _get_pages_from_text(
        cls,
        text: str,
    ) -> List[int]:
        """
        Find all internal page markers contained in a chunk.

        Example:

            <!-- PAGE:20 -->
            Revenue increased...

            <!-- PAGE:21 -->
            Revenue continued...

        returns:

            [20, 21]
        """

        pages = [
            int(match.group(1))
            for match in cls.INTERNAL_PAGE_REGEX.finditer(
                text
            )
        ]

        return sorted(set(pages))

    # ------------------------------------------------------------------
    # REMOVE INTERNAL PAGE MARKERS
    # ------------------------------------------------------------------

    @classmethod
    def _remove_page_markers(
        cls,
        text: str,
    ) -> str:
        """
        Remove internal page markers before storing
        the final chunk in ChromaDB.
        """

        return cls.INTERNAL_PAGE_REGEX.sub(
            "",
            text,
        ).strip()

    # ------------------------------------------------------------------
    # SPLIT LARGE SECTION
    # ------------------------------------------------------------------

    def _split_section(
        self,
        section: Dict,
    ) -> List[Dict]:
        """
        Split a section if it exceeds max_chunk_size.

        Page markers remain inside the text while splitting,
        allowing each resulting chunk to get its own accurate
        page_start/page_end values.
        """

        text = section["text"]

        # --------------------------------------------------------------
        # Section is small enough.
        # --------------------------------------------------------------

        if len(text) <= self.max_chunk_size:
            return [section]

        # --------------------------------------------------------------
        # Section is too large.
        # --------------------------------------------------------------

        sub_chunks = self.fallback_splitter.split_text(
            text
        )

        results = []

        for sub_chunk in sub_chunks:

            pages = self._get_pages_from_text(
                sub_chunk
            )

            results.append({
                "text": sub_chunk,
                "pages": pages,
                "headers": section["headers"],
            })

        return results

    # ------------------------------------------------------------------
    # MAIN FUNCTION
    # ------------------------------------------------------------------

    def chunk_document(
        self,
        full_markdown: str,
        source: str = "unknown",
    ) -> List[Dict]:
        """
        Create Markdown-aware chunks.

        Example returned chunk:

            {
                "text": "Revenue increased...",
                "metadata": {
                    "source": "annual_report.md",
                    "page_start": 20,
                    "page_end": 21,
                    "header_1": "Financial Performance",
                    "header_2": "Revenue"
                }
            }
        """

        if not full_markdown.strip():
            logger.warning(
                "Markdown document is empty"
            )
            return []

        sections = self._parse_sections(
            full_markdown
        )

        final_chunks: List[Dict] = []

        for section in sections:

            split_sections = self._split_section(
                section
            )

            for split_section in split_sections:

                raw_text = split_section["text"]

                clean_text = (
                    self._remove_page_markers(
                        raw_text
                    )
                )

                if not clean_text:
                    continue

                pages = split_section.get(
                    "pages",
                    [],
                )

                page_metadata = (
                    self.get_page_range(
                        pages
                    )
                )

                metadata = {
                    "source": os.path.basename(
                        source
                    ),
                    **page_metadata,
                }

                # ------------------------------------------------------
                # Add heading hierarchy.
                # ------------------------------------------------------

                headers = split_section.get(
                    "headers",
                    {},
                )

                for level, heading in headers.items():

                    if heading:
                        metadata[
                            f"header_{level}"
                        ] = heading

                final_chunks.append({
                    "text": clean_text,
                    "metadata": metadata,
                })

        logger.info(
            "Created %d chunks from %s",
            len(final_chunks),
            source,
        )

        return final_chunks


# ----------------------------------------------------------------------
# TEST / MANUAL EXECUTION
# ----------------------------------------------------------------------

if __name__ == "__main__":

    PROJECT_ROOT = Path(settings.PROJECT_ROOT)

    sample_md_path = (
        PROJECT_ROOT
        / settings.MARKDOWN_DIR
        / "sample.md"
    )


    if not os.path.isfile(sample_md_path):
        raise FileNotFoundError(
            f"Markdown file not found: {sample_md_path}"
        )

    with open(
        sample_md_path,
        "r",
        encoding="utf-8",
    ) as file:
        markdown = file.read()

    chunker = FinTechMarkdownChunker()

    chunks = chunker.chunk_document(
        markdown,
        source=sample_md_path,
    )

    print(
        f"\n--- Total Chunks: {len(chunks)} ---"
    )

    for index, chunk in enumerate(
        chunks[:10],
        start=1,
    ):
        metadata = chunk["metadata"]

        print(f"\n[Chunk {index}]")

        print(
            f"Page Start: "
            f"{metadata['page_start']}"
        )

        print(
            f"Page End: "
            f"{metadata['page_end']}"
        )

        print(
            f"Section: "
            f"{metadata.get('header_1', '')}"
        )

        print(
            f"Subsection: "
            f"{metadata.get('header_2', '')}"
        )

        print(
            f"Sub-subsection: "
            f"{metadata.get('header_3', '')}"
        )

        print("\nText:")
        print(chunk["text"])

        print("-" * 60)
