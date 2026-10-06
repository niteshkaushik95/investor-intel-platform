from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ------------------------------------------------------------------
# FINANCIAL PERIOD ENUM
# ------------------------------------------------------------------

class FinancialPeriod(str, Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    H1 = "H1"
    H2 = "H2"
    FY = "FY"
    YTD = "YTD"
    TTM = "TTM"
    OTHER = "OTHER"


# ------------------------------------------------------------------
# BASE EXTRACTION / PROVENANCE
# ------------------------------------------------------------------

class BaseExtraction(BaseModel):
    """
    Common contract for all LLM extractions.

    The LLM identifies:
    1. Whether the requested information is present.
    2. Which retrieved chunk contains the evidence.

    Page numbers, section names, snippets, etc. are NOT generated
    by the LLM. They are resolved deterministically from ChromaDB
    metadata by the application.
    """

    is_present: bool = Field(
        ...,
        description=(
            "True only when the requested metric or insight is "
            "explicitly supported by the provided context. "
            "False when it is unavailable."
        ),
    )

    evidence_chunk_id: Optional[str] = Field(
        default=None,
        description=(
            "Exact CHUNK_ID from the provided context containing "
            "the evidence. Required when is_present is True. "
            "Must be null when is_present is False."
        ),
    )

    @model_validator(mode="after")
    def validate_provenance(self):
        if self.is_present and not self.evidence_chunk_id:
            raise ValueError(
                "evidence_chunk_id is required when is_present=True"
            )

        if not self.is_present and self.evidence_chunk_id is not None:
            raise ValueError(
                "evidence_chunk_id must be null when is_present=False"
            )

        return self


# ------------------------------------------------------------------
# QUANTITATIVE FINANCIAL METRIC
# ------------------------------------------------------------------

class ExtractedMetric(BaseExtraction):
    """
    Strict schema for quantitative financial metrics.
    """

    numerical_value: Optional[Decimal] = Field(
        default=None,
        description=(
            "Exact financial value normalized to a numeric value. "
            "Do not include currency symbols or thousands separators. "
            "Preserve the value represented in the source document."
        ),
    )

    currency: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=3,
        description=(
            "Three-letter ISO currency code, such as USD, EUR, or INR. "
            "Null when currency is not explicitly available."
        ),
    )

    unit: Optional[str] = Field(
        default=None,
        description=(
            "Unit or scale of the value, such as absolute, thousands, "
            "millions, billions, %, or employees."
        ),
    )

    year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        description=(
            "Four-digit financial/reporting year associated with "
            "the extracted value."
        ),
    )

    period: Optional[FinancialPeriod] = Field(
        default=None,
        description=(
            "Financial reporting period associated with the value."
        ),
    )

    @model_validator(mode="after")
    def validate_metric_data(self):
        if not self.is_present:
            return self

        if self.numerical_value is None:
            raise ValueError(
                "numerical_value is required when is_present=True"
            )

        if self.year is None:
            raise ValueError(
                "year is required when is_present=True"
            )

        return self


# ------------------------------------------------------------------
# QUALITATIVE INSIGHT
# ------------------------------------------------------------------

class ExtractedInsight(BaseExtraction):
    """
    Schema for qualitative financial information such as:
    risks, growth drivers, strategic commentary, etc.
    """

    content_text: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "Concise factual summary of the requested insight, "
            "based only on the provided context."
        ),
    )

    @model_validator(mode="after")
    def validate_insight_data(self):
        if not self.is_present:
            return self

        if not self.content_text:
            raise ValueError(
                "content_text is required when is_present=True"
            )

        return self