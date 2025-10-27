"""Pydantic schemas for API request/response validation"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel, ConfigDict


class AnalysisJobCreateResponse(BaseModel):
    """Response schema when creating a new analysis job"""
    
    job_id: uuid.UUID
    

class AnalysisJobStatusResponse(BaseModel):
    """Response schema for job status and results"""
    
    id: uuid.UUID
    status: str
    created_at: datetime
    original_image_url: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)

