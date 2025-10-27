"""SQLAlchemy ORM models"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.database import Base


class AnalysisJob(Base):
    """Model for storing ad analysis jobs"""
    
    __tablename__ = "analysis_jobs"
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    status = Column(
        String,
        default="processing",
        index=True,
        nullable=False
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    original_image_url = Column(String, nullable=True)
    results = Column(JSONB, nullable=True)
    
    def __repr__(self):
        return f"<AnalysisJob(id={self.id}, status={self.status})>"

