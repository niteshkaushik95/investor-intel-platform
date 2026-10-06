# FIX 1: Removed unused 'Float' import
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Enum, Text, Numeric
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
    PENDING = "pending"       # Added for the atomic lock
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"       # Must be explicitly added
    FAILED = "failed"


# --- Table 1: Global Document Tracking ---

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    
    original_filename = Column(String, nullable=False)                         
    system_filename = Column(String, unique=True, nullable=False, index=True)  
    
    raw_pdf_path = Column(String, nullable=False)
    processed_md_path = Column(String)
    
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PROCESSING, nullable=False)
    file_hash = Column(String, index=True, nullable=True) 

    document_type = Column(Enum(DocumentType), default=DocumentType.UNKNOWN, nullable=False, index=True)
    report_year = Column(Integer, nullable=True, index=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    metrics = relationship("FinancialMetric", back_populates="document", cascade="all, delete")
    insights = relationship("DocumentInsight", back_populates="document", cascade="all, delete")

# --- Table 2: Extracted Financial Metrics ---

class FinancialMetric(Base):
    __tablename__ = "financial_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String, nullable=False, index=True) 
    
    numerical_value = Column(Numeric(20, 4), nullable=False)          
    currency = Column(String(3), nullable=True)          
    unit = Column(String, nullable=True)                 
    
    year = Column(Integer, nullable=False, index=True)       
    
    period = Column(String, nullable=True, index=True)
    period_description = Column(String, nullable=True)

    # Data Lineage
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)       
    section_name = Column(String, nullable=True)             
    source_snippet = Column(Text, nullable=True)             
    
    # FIX 3: Enforced mandatory provenance (nullable=False)
    chunk_id = Column(String, nullable=False, index=True)

    extracted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    document = relationship("Document", back_populates="metrics")
    
# --- Table 3: Qualitative Insights ---

class DocumentInsight(Base):
    __tablename__ = "document_insights"

    id = Column(Integer, primary_key=True, index=True)
    insight_name = Column(String, nullable=False, index=True)
    content_text = Column(Text, nullable=False)               

    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)             
    section_name = Column(String, nullable=True)             
    source_snippet = Column(Text, nullable=True)
    
    # FIX 3: Enforced mandatory provenance (nullable=False)
    chunk_id = Column(String, nullable=False, index=True)
    
    extracted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    document = relationship("Document", back_populates="insights")