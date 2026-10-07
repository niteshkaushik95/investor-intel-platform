import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.config import settings, METRIC_EXTRACTION_CONFIG

from app.models import (
    Document,
    DocumentStatus,
    FinancialMetric,
    DocumentInsight,
)

from app.utils.pdf_parser import convert_pdf_to_markdown
from app.services.chunker import FinTechMarkdownChunker
from app.services.vector_store import vector_store
from app.services.extractor import extract_metric_with_llm


logger = logging.getLogger(__name__)


def process_document_task(
    document_id: int,
    pdf_path: str,
    filename: str,
) -> None:
    """
    Master background worker for the FinTech document pipeline.

    Pipeline:

        PDF
          ↓
        Markdown
          ↓
        Page-aware chunking
          ↓
        ChromaDB
          ↓
        Per-metric RAG retrieval
          ↓
        LLM extraction with retries
          ↓
        Provenance validation
          ↓
        SQL persistence

    Important design decisions:

    - RAG retrieval happens once per metric.
    - Only the LLM call is retried.
    - Every extracted value must reference a retrieved chunk.
    - Page/section provenance comes from Chroma metadata, not the LLM.
    - SQL commits happen per successful metric.
    - Fatal pipeline failures clean up ChromaDB.
    """

    db: Session = SessionLocal()

    try:

        # ==============================================================
        # 1. ATOMIC DOCUMENT LOCK
        # ==============================================================

        updated_count = (
            db.query(Document)
            .filter(
                Document.id == document_id,
                Document.status.in_(
                    [
                        DocumentStatus.PENDING,
                        DocumentStatus.FAILED,
                    ]
                ),
            )
            .update(
                {
                    "status": DocumentStatus.PROCESSING
                },
                synchronize_session=False,
            )
        )

        if updated_count == 0:
            logger.warning(
                "Document %s is already processing, completed, "
                "or does not exist.",
                document_id,
            )
            return

        db.commit()

        doc = (
            db.query(Document)
            .filter(Document.id == document_id)
            .first()
        )

        if not doc:
            raise RuntimeError(
                f"Document {document_id} disappeared after locking."
            )

        logger.info(
            "Started processing Document %s",
            document_id,
        )

        # ==============================================================
        # 2. IDEMPOTENT SQL CLEANUP
        # ==============================================================

        db.query(FinancialMetric).filter(
            FinancialMetric.document_id == document_id
        ).delete(
            synchronize_session=False
        )

        db.query(DocumentInsight).filter(
            DocumentInsight.document_id == document_id
        ).delete(
            synchronize_session=False
        )

        db.commit()

        # ==============================================================
        # 3. PDF → MARKDOWN
        # ==============================================================

        logger.info(
            "Converting PDF to Markdown for Document %s",
            document_id,
        )

        md_filename = f"{Path(filename).stem}.md"

        md_path = (
            Path(settings.PROJECT_ROOT)
            / settings.MARKDOWN_DIR
            / md_filename
        )

        full_markdown = convert_pdf_to_markdown(
            pdf_path=pdf_path,
            output_path=str(md_path),
        )

        if not full_markdown or not full_markdown.strip():
            raise ValueError(
                f"PDF conversion produced empty Markdown "
                f"for document {document_id}"
            )

        # ==============================================================
        # 4. SMART CHUNKING
        # ==============================================================

        logger.info(
            "Chunking Document %s",
            document_id,
        )

        chunker = FinTechMarkdownChunker()

        chunks = chunker.chunk_document(
            full_markdown=full_markdown,
            source=filename,
        )

        if not chunks:
            raise ValueError(
                f"No chunks generated for document {document_id}"
            )

        # ==============================================================
        # 5. STORE CHUNKS IN CHROMA
        # ==============================================================

        logger.info(
            "Storing %d chunks in ChromaDB for Document %s",
            len(chunks),
            document_id,
        )

        vector_store.store_chunks(
            document_id=document_id,
            chunks=chunks,
        )

        # ==============================================================
        # 6. TARGETED METRIC EXTRACTION
        # ==============================================================

        total_metrics = len(
            METRIC_EXTRACTION_CONFIG
        )

        successful_metrics = 0
        missing_metrics = 0
        failed_metrics = 0

        for metric_key, config in (
            METRIC_EXTRACTION_CONFIG.items()
        ):

            metric_name = config["name"]

            logger.info(
                "Processing metric: %s",
                metric_name,
            )

            # ----------------------------------------------------------
            # 6A. RETRIEVE RAG CONTEXT ONCE
            # ----------------------------------------------------------

            top_k = config.get(
                "top_k",
                (
                    10
                    if config["type"] == "insight"
                    else settings.DEFAULT_TOP_K
                ),
            )

            try:
                results = (
                    vector_store.search_similar_chunks(
                        query=config["query"],
                        top_k=top_k,
                        document_id=document_id,
                    )
                )
            except Exception:
                logger.exception(
                    "RAG retrieval failed for metric %s",
                    metric_name,
                )
                failed_metrics += 1
                continue

            if (
                not results
                or not results.get("ids")
                or not results["ids"][0]
            ):
                logger.info(
                    "No relevant chunks found for %s. "
                    "Treating metric as not found.",
                    metric_name,
                )

                missing_metrics += 1
                continue

            # ----------------------------------------------------------
            # 6B. BUILD CONTEXT + PROVENANCE MAP
            # ----------------------------------------------------------

            context_blocks = []
            chunk_metadata_map = {}

            result_ids = results["ids"][0]
            result_documents = results["documents"][0]
            result_metadatas = results["metadatas"][0]

            for index, chunk_id in enumerate(
                result_ids
            ):

                chunk_text = result_documents[index]
                chunk_meta = result_metadatas[index]

                chunk_metadata_map[chunk_id] = {
                    "text": chunk_text,
                    "meta": chunk_meta,
                }

                context_blocks.append(
                    f"--- [CHUNK_ID: {chunk_id}] ---\n"
                    f"{chunk_text}"
                )

            full_context = "\n\n".join(
                context_blocks
            )

            # ----------------------------------------------------------
            # 6C. LLM EXTRACTION WITH RETRIES
            #
            # Retrieval is intentionally NOT repeated.
            # ----------------------------------------------------------

            metric_extracted = False

            for attempt in range(
                1,
                settings.EXTRACTION_MAX_RETRIES + 1,
            ):

                try:

                    logger.info(
                        "Extracting %s "
                        "(attempt %d/%d)",
                        metric_name,
                        attempt,
                        settings.EXTRACTION_MAX_RETRIES,
                    )

                    llm_response = (
                        extract_metric_with_llm(
                            target_name=metric_name,
                            context=full_context,
                            extraction_type=config["type"],
                        )
                    )

                    if llm_response is None:
                        raise ValueError(
                            "LLM returned no response."
                        )

                    # --------------------------------------------------
                    # Explicit presence decision
                    # --------------------------------------------------

                    if not llm_response.is_present:

                        logger.info(
                            "Metric %s is not present "
                            "in the retrieved evidence.",
                            metric_name,
                        )

                        missing_metrics += 1
                        metric_extracted = True
                        break

                    # --------------------------------------------------
                    # Validate evidence chunk
                    # --------------------------------------------------

                    evidence_chunk_id = (
                        llm_response.evidence_chunk_id
                    )

                    if not evidence_chunk_id:
                        raise ValueError(
                            "LLM did not provide evidence_chunk_id."
                        )

                    if (
                        evidence_chunk_id
                        not in chunk_metadata_map
                    ):
                        raise ValueError(
                            "LLM returned an invalid "
                            f"evidence_chunk_id: "
                            f"{evidence_chunk_id}"
                        )

                    evidence = (
                        chunk_metadata_map[
                            evidence_chunk_id
                        ]
                    )

                    meta = evidence["meta"]

                    # --------------------------------------------------
                    # FINANCIAL METRIC
                    # --------------------------------------------------

                    if (
                        config["type"]
                        == "financial_metric"
                    ):

                        if (
                            llm_response.numerical_value
                            is None
                        ):
                            raise ValueError(
                                "Missing numerical_value."
                            )

                        if (
                            llm_response.year
                            is None
                        ):
                            raise ValueError(
                                "Missing year."
                            )

                        # ----------------------------------------------
                        # Validate report year.
                        # ----------------------------------------------

                        if (
                            doc.report_year is not None
                            and llm_response.year
                            != doc.report_year
                        ):
                            raise ValueError(
                                f"Extracted year "
                                f"{llm_response.year} does not "
                                f"match document report year "
                                f"{doc.report_year}."
                            )

                        # ----------------------------------------------
                        # Validate required financial context.
                        # ----------------------------------------------

                        if not llm_response.unit:
                            raise ValueError(
                                "Missing financial unit."
                            )

                        if not llm_response.period:
                            raise ValueError(
                                "Missing financial period."
                            )

                        db_metric = FinancialMetric(
                            document_id=document_id,
                            metric_name=metric_name,

                            numerical_value=(
                                llm_response.numerical_value
                            ),

                            currency=(
                                llm_response.currency
                            ),

                            unit=(
                                llm_response.unit
                            ),

                            year=(
                                llm_response.year
                            ),

                            period=(
                                llm_response.period
                            ),

                            page_start=(
                                meta.get("page_start")
                            ),

                            page_end=(
                                meta.get("page_end")
                            ),

                            section_name=(
                                meta.get("header_1")
                            ),

                            source_snippet=(
                                evidence["text"][:1000]
                            ),

                            chunk_id=evidence_chunk_id,
                        )

                        db.add(db_metric)

                    # --------------------------------------------------
                    # QUALITATIVE INSIGHT
                    # --------------------------------------------------

                    elif (
                        config["type"]
                        == "insight"
                    ):

                        if not (
                            llm_response.content_text
                            and llm_response.content_text.strip()
                        ):
                            raise ValueError(
                                "Missing insight content."
                            )

                        db_insight = DocumentInsight(
                            document_id=document_id,

                            insight_name=metric_name,

                            content_text=(
                                llm_response.content_text
                            ),

                            page_start=(
                                meta.get("page_start")
                            ),

                            page_end=(
                                meta.get("page_end")
                            ),

                            section_name=(
                                meta.get("header_1")
                            ),

                            source_snippet=(
                                evidence["text"][:1000]
                            ),

                            chunk_id=evidence_chunk_id,
                        )

                        db.add(db_insight)

                    else:
                        raise ValueError(
                            f"Unknown extraction type: "
                            f"{config['type']}"
                        )

                    # --------------------------------------------------
                    # Persist successful metric immediately.
                    # --------------------------------------------------

                    db.commit()

                    successful_metrics += 1
                    metric_extracted = True

                    logger.info(
                        "Successfully extracted %s",
                        metric_name,
                    )

                    break

                except Exception as metric_error:

                    db.rollback()

                    if attempt < settings.EXTRACTION_MAX_RETRIES:

                        logger.warning(
                            "Attempt %d failed for %s: %s. "
                            "Retrying LLM extraction.",
                            attempt,
                            metric_name,
                            metric_error,
                        )

                    else:

                        logger.exception(
                            "All %d LLM attempts failed "
                            "for %s.",
                            settings.EXTRACTION_MAX_RETRIES,
                            metric_name,
                        )

            if not metric_extracted:
                failed_metrics += 1

        # ==============================================================
        # 7. FINAL DOCUMENT STATUS
        # ==============================================================

        doc.processed_md_path = str(
            md_path
        )

        if failed_metrics == 0:

            # Missing metrics are not pipeline failures.
            doc.status = (
                DocumentStatus.COMPLETED
            )

            logger.info(
                "Document %s completed. "
                "Successful=%d, Missing=%d.",
                document_id,
                successful_metrics,
                missing_metrics,
            )

        elif successful_metrics > 0:

            doc.status = (
                DocumentStatus.PARTIAL
            )

            logger.warning(
                "Document %s partially completed. "
                "Successful=%d, Missing=%d, Failed=%d.",
                document_id,
                successful_metrics,
                missing_metrics,
                failed_metrics,
            )

        else:

            doc.status = (
                DocumentStatus.FAILED
            )

            logger.error(
                "All metric extractions failed "
                "for Document %s.",
                document_id,
            )

        db.commit()

    # ==============================================================
    # FATAL PIPELINE FAILURE
    # ==============================================================

    except Exception:

        logger.exception(
            "Fatal pipeline failure for Document %s",
            document_id,
        )

        db.rollback()

        # ----------------------------------------------------------
        # ChromaDB is not transactional.
        #
        # SQL rollback cannot undo vector insertion.
        # ----------------------------------------------------------

        try:

            vector_store.delete_document(
                document_id
            )

            logger.info(
                "Removed ChromaDB chunks for failed "
                "Document %s.",
                document_id,
            )

        except Exception:

            logger.exception(
                "Failed to remove ChromaDB chunks "
                "for Document %s.",
                document_id,
            )

        # ----------------------------------------------------------
        # Mark document failed.
        # ----------------------------------------------------------

        try:

            failed_doc = (
                db.query(Document)
                .filter(
                    Document.id == document_id
                )
                .first()
            )

            if failed_doc:

                failed_doc.status = (
                    DocumentStatus.FAILED
                )

                db.commit()

        except Exception:

            logger.exception(
                "Failed to mark Document %s "
                "as FAILED.",
                document_id,
            )

        raise

    finally:

        db.close()
