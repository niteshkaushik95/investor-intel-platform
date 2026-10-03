from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime, timezone
from .database import Base

# This class translates directly into a SQL table named "financial_metrics"
class FinancialMetric(Base):
    __tablename__ = "financial_metrics"

    # Define the columns of the table
    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String, index=True)
    year = Column(Integer, index=True)
    
    # Financial Data
    total_revenue = Column(Float, nullable=True)
    net_income = Column(Float, nullable=True)
    
    # AI Extracted Summaries
    top_risk_factors = Column(String, nullable=True)
    top_growth_drivers = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))