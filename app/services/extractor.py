import logging
from typing import Optional, Union

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.schemas import ExtractedMetric, ExtractedInsight


logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM_PROMPT = """
You are an elite, audit-level FinTech AI assistant.

Your job is to extract ONE specific financial metric or insight
from the provided document chunks.

CRITICAL RULES:

1. TRUTH & HALLUCINATION
   - Use ONLY the provided evidence chunks.
   - Never use outside knowledge.
   - Never guess or infer missing information.

2. NO EVIDENCE = NO EXTRACTION
   - If the requested information is not explicitly supported
     by the provided chunks, set `is_present` to false.
   - When `is_present` is false, all extraction-specific fields
     must be null.
   - `evidence_chunk_id` must also be null.

3. PROVENANCE IS MANDATORY
   - Each chunk is identified by a marker such as:
     `--- [CHUNK_ID: doc_45_chunk_2] ---`
   - When `is_present` is true, copy the EXACT chunk ID containing
     the evidence into `evidence_chunk_id`.
   - Never invent or modify a chunk ID.

4. FINANCIAL METRICS
   - Extract the exact numerical value supported by the source.
   - Do not perform calculations.
   - Do not convert currencies.
   - Preserve the source scale using the `unit` field.
   - Extract year and period only when supported by the evidence.

5. QUALITATIVE INSIGHTS
   - Return only information explicitly supported by the evidence.
   - Do not introduce external interpretation or facts.

6. OUTPUT
   - Follow the provided structured output schema exactly.
"""


def get_llm() -> ChatGroq:
    """
    Create the configured Groq LLM client.
    """

    return ChatGroq(
        temperature=0.0,
        model_name=settings.LLM_MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def extract_metric_with_llm(
    target_name: str,
    context: str,
    extraction_type: str,
) -> Optional[Union[ExtractedMetric, ExtractedInsight]]:
    """
    Extract one configured metric or insight from retrieved RAG context.

    The LLM is responsible only for extraction.
    Provenance metadata is resolved by the calling service using
    the returned evidence_chunk_id.
    """

    if not context or not context.strip():
        logger.warning(
            "Empty context supplied for target '%s'",
            target_name,
        )
        return None

    if extraction_type == "financial_metric":
        schema = ExtractedMetric

    elif extraction_type == "insight":
        schema = ExtractedInsight

    else:
        logger.error(
            "Unknown extraction_type: %s",
            extraction_type,
        )
        return None

    structured_llm = get_llm().with_structured_output(schema)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                EXTRACTION_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
TARGET TO EXTRACT:
{target_name}

EXTRACTION TYPE:
{extraction_type}

EVIDENCE CHUNKS:
{context}
""",
            ),
        ]
    )

    chain = prompt | structured_llm

    try:
        return chain.invoke(
            {
                "target_name": target_name,
                "extraction_type": extraction_type,
                "context": context,
            }
        )

    except Exception:
        logger.exception(
            "LLM extraction failed for target '%s'",
            target_name,
        )
        raise