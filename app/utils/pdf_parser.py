import logging
from pathlib import Path

import pymupdf4llm

logger = logging.getLogger(__name__)


def convert_pdf_to_markdown(
    pdf_path: str,
    output_path: str,
) -> str:
    """
    Converts a PDF to Markdown while preserving page boundaries.

    Page markers are consumed by the chunker to attach
    page metadata to ChromaDB chunks.
    """

    pdf_file = Path(pdf_path)
    out_path = Path(output_path)

    if not pdf_file.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    logger.info(
        "Extracting Markdown from: %s",
        pdf_file,
    )

    try:
        page_data = pymupdf4llm.to_markdown(
            str(pdf_file),
            page_chunks=True,
        )

        if not page_data:
            raise ValueError(
                f"No pages extracted from PDF: {pdf_path}"
            )

        full_markdown_blocks = []

        for index, page in enumerate(page_data, start=1):
            page_text = page.get("text", "").strip()

            if not page_text:
                logger.warning(
                    "No text extracted from page %d of %s",
                    index,
                    pdf_path,
                )

            page_block = (
                f"--- [START OF PAGE {index}] ---\n"
                f"{page_text}\n"
                f"--- [END OF PAGE {index}] ---"
            )

            full_markdown_blocks.append(page_block)

        full_markdown = "\n\n".join(
            full_markdown_blocks
        )

        out_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with out_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            f.write(full_markdown)

        logger.info(
            "Successfully converted PDF to Markdown. "
            "Saved to %s",
            out_path,
        )

        return full_markdown

    except Exception:
        logger.exception(
            "Failed to convert PDF to Markdown: %s",
            pdf_path,
        )
        raise