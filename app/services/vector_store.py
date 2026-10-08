import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import json # Add to top of file

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings


logger = logging.getLogger(__name__)


class FinTechVectorStore:
    """
    Handles storing and searching document chunks in ChromaDB.

    Responsibilities:
    - Create and manage persistent ChromaDB storage.
    - Generate embeddings using Chroma's default embedding function.
    - Store chunks and their metadata safely.
    - Remove stale chunks when a document is re-ingested.
    - Search for semantically similar chunks.
    """

    def __init__(
        self,
        collection_name: str = "fintech_documents",
        batch_size: int = settings.CHROMA_BATCH_SIZE,
    ):
        """
        Initialize the ChromaDB client.

        Args:
            collection_name:
                Name of the ChromaDB collection.

            batch_size:
                Number of chunks to insert into ChromaDB
                in a single batch.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )

        self.collection_name = collection_name
        self.batch_size = batch_size

        # --------------------------------------------------------------
        # 1. Create ChromaDB directory if it doesn't exist.
        # --------------------------------------------------------------

        chroma_path = Path(settings.CHROMA_DIR)

        chroma_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------------
        # 2. Initialize persistent ChromaDB client.
        #
        # ChromaSettings is renamed so it doesn't conflict with
        # our application's `settings`.
        # --------------------------------------------------------------

        self.client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(
                anonymized_telemetry=False
            ),
        )

        logger.info(
            "ChromaDB initialized at %s",
            chroma_path,
        )

        logger.info(
            "Using Chroma's default embedding function"
        )

    # ------------------------------------------------------------------
    # GET COLLECTION
    # ------------------------------------------------------------------

    def _get_collection(self):
        """
        Get the ChromaDB collection.

        No embedding function is explicitly provided, so Chroma
        uses its configured default embedding function.
        """

        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": (
                    "Financial document chunks for RAG"
                )
            },
        )

    # ------------------------------------------------------------------
    # VALIDATE CHUNK
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_chunk(
        chunk: Dict,
        index: int,
    ) -> None:
        """
        Validate the structure of a chunk before sending it to Chroma.
        """

        if not isinstance(chunk, dict):
            raise TypeError(
                f"Chunk {index} must be a dictionary"
            )

        if "text" not in chunk:
            raise ValueError(
                f"Chunk {index} is missing 'text'"
            )

        if "metadata" not in chunk:
            raise ValueError(
                f"Chunk {index} is missing 'metadata'"
            )

        if not isinstance(chunk["text"], str):
            raise TypeError(
                f"Chunk {index} 'text' must be a string"
            )

        if not chunk["text"].strip():
            raise ValueError(
                f"Chunk {index} contains empty text"
            )

        if not isinstance(
            chunk["metadata"],
            dict,
        ):
            raise TypeError(
                f"Chunk {index} 'metadata' must be a dictionary"
            )

    # ------------------------------------------------------------------
    # DELETE DOCUMENT
    # ------------------------------------------------------------------

    def delete_document(
        self,
        document_id: int,
    ) -> None:
        """
        Delete all chunks belonging to a document.

        This prevents stale chunks from remaining in ChromaDB when
        a document is re-ingested and produces fewer or different
        chunks.
        """

        collection = self._get_collection()

        logger.info(
            "Deleting existing chunks for document %s",
            document_id,
        )

        collection.delete(
            where={
                "document_id": document_id
            }
        )

        logger.info(
            "Finished deleting existing chunks for document %s",
            document_id,
        )

    # ------------------------------------------------------------------
    # STORE CHUNKS
    # ------------------------------------------------------------------

    def store_chunks(
        self,
        document_id: int,
        chunks: List[Dict],
    ) -> None:
        """
        Generate embeddings and store document chunks in ChromaDB.

        If the document already exists, its old chunks are deleted first.
        This prevents stale chunks from previous ingestion runs.
        """

        if not chunks:
            logger.warning(
                "No chunks provided for document %s. "
                "Skipping ChromaDB insertion.",
                document_id,
            )
            return

        # --------------------------------------------------------------
        # 1. Validate all chunks before modifying ChromaDB.
        # --------------------------------------------------------------

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            self._validate_chunk(
                chunk,
                index,
            )

        try:
            collection = self._get_collection()

            # ----------------------------------------------------------
            # 2. Remove old chunks.
            # ----------------------------------------------------------

            self.delete_document(
                document_id
            )

            # ----------------------------------------------------------
            # 3. Prepare data for ChromaDB.
            # ----------------------------------------------------------

            texts: List[str] = []
            metadatas: List[Dict] = []
            ids: List[str] = []

            # One timestamp for the entire ingestion operation.
            current_time = datetime.now(
                timezone.utc
            ).isoformat()

            # ChromaDB supports these basic metadata value types.
            valid_types = (
                str,
                int,
                float,
                bool,
            )

            for index, chunk in enumerate(
                chunks,
                start=1,
            ):
                texts.append(
                    chunk["text"]
                )

                metadata = chunk["metadata"].copy()

                # Add relational document ID.
                metadata["document_id"] = document_id

                # Add ingestion timestamp.
                metadata["ingested_at"] = current_time

                # ------------------------------------------------------
                # Validate and clean metadata.
                #
                # None values are skipped because ChromaDB doesn't
                # accept them as metadata values.
                #
                # Invalid types raise an error instead of being
                # silently discarded.
                # ------------------------------------------------------

                clean_metadata = {}

                for key, value in metadata.items():

                    if value is None:
                        continue

                    # NEW: Convert lists and dicts to JSON strings for ChromaDB
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value)

                    if not isinstance(
                        value,
                        valid_types,
                    ):
                        raise TypeError(
                            f"Chunk {index} metadata field "
                            f"'{key}' has unsupported type: "
                            f"{type(value).__name__}"
                        )

                    clean_metadata[key] = value

                metadatas.append(
                    clean_metadata
                )

                # Deterministic ID.
                #
                # Example:
                #
                # doc_45_chunk_1
                # doc_45_chunk_2
                # doc_45_chunk_3
                #

                ids.append(
                    f"doc_{document_id}_chunk_{index}"
                )

            # ----------------------------------------------------------
            # 4. Insert chunks in batches.
            # ----------------------------------------------------------

            total_chunks = len(texts)

            for start in range(
                0,
                total_chunks,
                self.batch_size,
            ):
                end = min(
                    start + self.batch_size,
                    total_chunks,
                )

                logger.info(
                    "Storing chunks %d-%d of %d "
                    "for document %s",
                    start + 1,
                    end,
                    total_chunks,
                    document_id,
                )

                collection.upsert(
                    documents=texts[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                )

            logger.info(
                "Successfully embedded and stored %d chunks "
                "for document %s",
                total_chunks,
                document_id,
            )

        except Exception:
            logger.exception(
                "Failed to store chunks in ChromaDB "
                "for document %s",
                document_id,
            )
            raise

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search_similar_chunks(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[int] = None,
    ):
        """
        Search ChromaDB for chunks semantically similar to a query.

        Args:
            query:
                User's natural-language question.

            top_k:
                Number of chunks to retrieve.

            document_id:
                Optional document ID.

                If provided, search only within that document.
                If None, search across all documents.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        try:
            collection = self._get_collection()

            logger.info(
                "Searching ChromaDB: query=%r, top_k=%d, "
                "document_id=%s",
                query,
                top_k,
                document_id,
            )

            query_kwargs = {
                "query_texts": [query],
                "n_results": top_k,
            }

            # ----------------------------------------------------------
            # Search only within one document when requested.
            # ----------------------------------------------------------

            if document_id is not None:
                query_kwargs["where"] = {
                    "document_id": document_id
                }

            results = collection.query(
                **query_kwargs
            )

            return results

        except Exception:
            logger.exception(
                "ChromaDB search failed for query: %r",
                query,
            )
            raise


# ----------------------------------------------------------------------
# GLOBAL VECTOR STORE
# ----------------------------------------------------------------------

vector_store = FinTechVectorStore()
