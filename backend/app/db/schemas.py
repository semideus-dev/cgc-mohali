"""Pydantic schemas for API request/response validation"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

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


# ============= MoodBoard Schemas =============

class MoodBoardCreate(BaseModel):
    """Request schema for creating a new moodboard"""
    user_id: str  # User ID for the moodboard owner
    brand_name: str
    brand_slogan: Optional[str] = None
    description: Optional[str] = None
    color_palette: Optional[List[str]] = None  # Array of color codes like ["#FF5733", "#33FF57"]
    images: Optional[List[str]] = None  # Array of image URLs
    prompt: Optional[str] = None  # Single prompt for all images


class MoodBoardUpdate(BaseModel):
    """Request schema for updating a moodboard"""
    brand_name: Optional[str] = None
    brand_slogan: Optional[str] = None
    description: Optional[str] = None
    color_palette: Optional[List[str]] = None
    images: Optional[List[str]] = None  # Array of image URLs
    prompt: Optional[str] = None  # Single prompt


class MoodBoardResponse(BaseModel):
    """Response schema for moodboard"""
    id: uuid.UUID
    user_id: str
    brand_name: str
    brand_slogan: Optional[str] = None
    description: Optional[str] = None
    color_palette: Optional[List[str]] = None
    images: Optional[List[str]] = None  # Array of image URLs
    prompt: Optional[str] = None  # Single prompt for all images
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============= MoodBoardCanvas Schemas =============

class MoodBoardCanvasCreate(BaseModel):
    """Request schema for creating a new moodboard canvas"""
    moodboard_id: uuid.UUID
    name: Optional[str] = "Untitled Canvas"
    canvas_urls: List[str]  # Array of canvas URLs in same order as moodboard images
    prompt: str
    generation_params: Optional[Dict[str, Any]] = None


class MoodBoardCanvasUpdate(BaseModel):
    """Request schema for updating a moodboard canvas"""
    name: Optional[str] = None
    is_favorite: Optional[bool] = None


class MoodBoardCanvasResponse(BaseModel):
    """Response schema for moodboard canvas"""
    id: uuid.UUID
    moodboard_id: uuid.UUID
    name: str
    canvas_urls: List[str]  # Array of canvas URLs in same order as moodboard images
    prompt: str
    is_favorite: bool
    generation_params: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

