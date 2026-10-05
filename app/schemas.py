from pydantic import BaseModel, Field
from typing import List, Optional
from decimal import Decimal
from enum import Enum # NEW IMPORT

# --- FINTECH PERIOD ENUM ---
class FinancialPeriod(str, Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    H1 = "H1"          # First Half
    H2 = "H2"          # Second Half
    FY = "FY"          # Full Year
    YTD = "YTD"        # Year to Date
    TTM = "TTM"        # Trailing Twelve Months
    OTHER = "OTHER"    # Fallback for weird edge cases


class ExtractedMetric(BaseModel):
    metric_name: str = Field(
        min_length=1, 
        description="Name of the financial metric, e.g., 'Total Revenue'."
    )
    numerical_value: Decimal = Field(
        description="Exact normalized financial value. Return as a decimal-compatible number, without currency symbols or commas."
    )
    currency: Optional[str] = Field(
        default=None, 
        min_length=3, 
        max_length=3, 
        description="3-letter ISO currency code, e.g., 'USD', 'EUR', 'INR'."
    )
    unit: Optional[str] = Field(
        default=None, 
        description="e.g., 'millions', '%', 'employees', 'absolute'."
    )
    year: int = Field(
        ge=1900, 
        le=2100, 
        description="The 4-digit financial year, e.g., 2026."
    )
    period_enum: FinancialPeriod = Field(
        description="The financial period. You MUST choose exactly one of the provided enum string values (e.g., 'Q1', 'FY', 'TTM')."
    )
    period_description: Optional[str] = Field(
        default=None,
        description=(
            "Original wording describing the reporting period, "
            "especially when period_enum is OTHER. "
            "Example: 'Nine months ended September 30, 2026'."
        )
    )
    page_number: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="The PDF page number where the information was found."
    )
    section_name: Optional[str] = Field(
        default=None, 
        description="The heading or section name."
    )
    source_snippet: str = Field(
        min_length=1,
        description="Exact verbatim text from the source proving this extraction. Do not summarize or modify it."
    )

class ExtractedInsight(BaseModel):
    insight_name: str = Field(min_length=1, description="e.g., 'Risk Factors'")
    content_text: str = Field(min_length=1, description="The detailed text or synthesized summary")
    page_number: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="The PDF page number where the information was found."
    )
    section_name: Optional[str] = Field(
        default=None, 
        description="The heading or section name."
    )
    source_snippet: str = Field(
        min_length=1,
        description="Exact verbatim text from the source proving this extraction. Do not summarize or modify it."
    )

class ExtractionResponse(BaseModel):
    metrics: List[ExtractedMetric]
    insights: List[ExtractedInsight]