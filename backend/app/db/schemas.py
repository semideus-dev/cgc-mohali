"""Pydantic schemas for API request/response validation"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel, ConfigDict, EmailStr


# ============= Job Creation and Upload =============

class UploadRequest(BaseModel):
    """Request schema for uploading image and prompt"""
    prompt: Optional[str] = None


class UploadResponse(BaseModel):
    """Response schema when uploading image"""
    job_id: uuid.UUID
    status: str
    original_image_url: str
    message: str


# ============= Analysis Responses =============

class OCRResults(BaseModel):
    """OCR extraction results"""
    raw_text: str
    confidence: float
    word_count: int
    char_count: int


class TextAnalysisResults(BaseModel):
    """Comprehensive text analysis results"""
    sentiment: Dict[str, Any]
    emotions: Dict[str, float]
    keywords: list[str]
    readability_score: float
    tone: str
    language: str
    entities: list[Dict[str, str]]


class ImageAnalysisResults(BaseModel):
    """Comprehensive image analysis results"""
    objects: Dict[str, Any]
    colors: Dict[str, Any]
    composition: Dict[str, Any]
    quality_metrics: Dict[str, float]
    visual_hierarchy: list[str]


class CritiqueResults(BaseModel):
    """Critique and recommendations"""
    overall_score: float
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    target_audience: str
    emotional_impact: str


class AnalysisResponse(BaseModel):
    """Response after analysis completion"""
    job_id: uuid.UUID
    status: str
    ocr_results: Optional[OCRResults] = None
    text_analysis: Optional[TextAnalysisResults] = None
    image_analysis: Optional[ImageAnalysisResults] = None


class CritiqueResponse(BaseModel):
    """Response after critique generation"""
    job_id: uuid.UUID
    status: str
    critique: Optional[CritiqueResults] = None
    master_prompt: Optional[str] = None


class GenerationResponse(BaseModel):
    """Response after image generation"""
    job_id: uuid.UUID
    status: str
    generated_image_url: Optional[str] = None
    master_prompt: Optional[str] = None


# ============= Legacy/Complete Results =============

class AnalysisJobCreateResponse(BaseModel):
    """Response schema when creating a new analysis job (legacy)"""
    job_id: uuid.UUID


class AnalysisJobStatusResponse(BaseModel):
    """Response schema for job status and complete results"""
    id: uuid.UUID
    user_id: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime
    original_image_url: Optional[str] = None
    user_prompt: Optional[str] = None
    ocr_results: Optional[Dict[str, Any]] = None
    text_analysis: Optional[Dict[str, Any]] = None
    image_analysis: Optional[Dict[str, Any]] = None
    critique: Optional[Dict[str, Any]] = None
    master_prompt: Optional[str] = None
    generated_image_url: Optional[str] = None
    results: Optional[Dict[str, Any]] = None  # Legacy field
    
    model_config = ConfigDict(from_attributes=True)

