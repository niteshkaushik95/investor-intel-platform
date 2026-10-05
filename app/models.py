from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Enum, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base
import enum

# --- Enums ---

class DocumentType(enum.Enum):
    ANNUAL_REPORT = "annual_report"
    QUARTER_REPORT = "quarter_report"
    INVESTOR_DECK = "investor_deck"
    UNKNOWN = "unknown"

class DocumentStatus(enum.Enum):
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


# --- Table 1: Global Document Tracking ---

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    
    # NEW: Store both the user's name and the system's unique name
    original_filename = Column(String, nullable=False)                         # e.g., 'report.pdf'
    system_filename = Column(String, unique=True, nullable=False, index=True)  # e.g., 'report_1696512345.pdf'
    
    raw_pdf_path = Column(String, nullable=False)
    processed_md_path = Column(String)
    
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PROCESSING, nullable=False)
    file_hash = Column(String, index=True, nullable=True) 

    # Global Context
    document_type = Column(Enum(DocumentType), default=DocumentType.UNKNOWN, nullable=False, index=True)
    report_year = Column(Integer, nullable=True, index=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    metrics = relationship("FinancialMetric", back_populates="document", cascade="all, delete")
    insights = relationship("DocumentInsight", back_populates="document", cascade="all, delete")

# --- Table 2: Quantitative Key-Value Data (Numbers) ---

class FinancialMetric(Base):
    __tablename__ = "financial_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String, nullable=False, index=True) 
    
    numerical_value = Column(Numeric(20, 4), nullable=False)          
    
    # --- NEW CONTEXT COLUMNS ---
    currency = Column(String(3), nullable=True)          # e.g., "USD", "EUR", "INR"
    unit = Column(String, nullable=True)                 # e.g., "%", "millions", "employees", "bps"
    
    year = Column(Integer, nullable=False, index=True)       
    period_enum = Column(String, nullable=False, index=True)
    period_description = Column(String, nullable=True)

    # Data Lineage
    page_number = Column(Integer, nullable=True)             
    section_name = Column(String, nullable=True)             
    source_snippet = Column(Text, nullable=True)             

    extracted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    document = relationship("Document", back_populates="metrics")
    
class DocumentInsight(Base):
    __tablename__ = "document_insights"

    id = Column(Integer, primary_key=True, index=True)
    insight_name = Column(String, nullable=False, index=True)
    content_text = Column(Text, nullable=False)               

    page_number = Column(Integer, nullable=True)             
    section_name = Column(String, nullable=True)             
    
    # FIX: Added snippet for auditability on qualitative synthesis
    source_snippet = Column(Text, nullable=True)
    
    extracted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    document = relationship("Document", back_populates="insights")