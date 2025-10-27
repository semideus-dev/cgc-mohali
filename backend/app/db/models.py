"""SQLAlchemy ORM models"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, func, Boolean, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.database import Base


class User(Base):
    """User model for authentication and tracking"""
    
    __tablename__ = "user"
    
    id = Column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(Text, nullable=False)
    email = Column(Text, nullable=False, unique=True, index=True)
    email_verified = Column(Boolean, default=False, nullable=False)
    image = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    accounts = relationship("Account", back_populates="user", cascade="all, delete-orphan")
    canvases = relationship("Canvas", back_populates="user", cascade="all, delete-orphan")
    analysis_jobs = relationship("AnalysisJob", back_populates="user", cascade="all, delete-orphan")
    moodboards = relationship("MoodBoard", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    """Session model for user authentication"""
    
    __tablename__ = "session"
    
    id = Column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    token = Column(Text, nullable=False, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    ip_address = Column(Text, nullable=True)
    user_agent = Column(Text, nullable=True)
    user_id = Column(Text, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class Account(Base):
    """Account model for OAuth providers"""
    
    __tablename__ = "account"
    
    id = Column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    account_id = Column(Text, nullable=False)
    provider_id = Column(Text, nullable=False)
    user_id = Column(Text, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    id_token = Column(Text, nullable=True)
    access_token_expires_at = Column(DateTime(timezone=True), nullable=True)
    refresh_token_expires_at = Column(DateTime(timezone=True), nullable=True)
    scope = Column(Text, nullable=True)
    password = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="accounts")


class Verification(Base):
    """Verification model for email/phone verification"""
    
    __tablename__ = "verification"
    
    id = Column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    identifier = Column(Text, nullable=False)
    value = Column(Text, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=True)


class Canvas(Base):
    """Canvas model for storing generated ad designs"""
    
    __tablename__ = "canvas"
    
    id = Column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(Text, nullable=False)
    url = Column(Text, nullable=True)
    prompt = Column(Text, nullable=False)
    user_id = Column(Text, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="canvases")


class AnalysisJob(Base):
    """Model for storing ad analysis jobs with comprehensive tracking"""
    
    __tablename__ = "analysis_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(Text, ForeignKey("user.id", ondelete="CASCADE"), nullable=True)
    status = Column(String, default="pending", index=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default = datetime.now(), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default = datetime.now(), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Input data
    original_image_url = Column(String, nullable=True)
    user_prompt = Column(Text, nullable=True)
    
    # Analysis results (stored as JSONB for flexibility)
    ocr_results = Column(JSONB, nullable=True)  # Text extraction results
    text_analysis = Column(JSONB, nullable=True)  # NLP analysis
    image_analysis = Column(JSONB, nullable=True)  # Vision analysis
    
    # Critique and generation
    critique = Column(JSONB, nullable=True)  # Comprehensive critique
    master_prompt = Column(Text, nullable=True)  # Generated prompt for image generation
    generated_image_url = Column(String, nullable=True)  # Final generated image
    
    # Legacy field for backward compatibility
    results = Column(JSONB, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="analysis_jobs")
    
    def __repr__(self):
        return f"<AnalysisJob(id={self.id}, status={self.status}, user_id={self.user_id})>"



class MoodBoard(Base):
    """MoodBoard model for storing brand inspiration with images and prompts"""
    
    __tablename__ = "moodboard"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(Text, ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    
    # Brand information
    brand_name = Column(Text, nullable=False)
    brand_slogan = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    color_palette = Column(JSONB, nullable=True)  # Array of color codes: ["#FF5733", "#33FF57"]
    
    # Images array and single prompt
    # images: Array of image URLs: ["https://...", "https://...", ...]
    images = Column(JSONB, nullable=True)  # Array of image URLs
    prompt = Column(Text, nullable=True)  # Single prompt for all images
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default = datetime.now(), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default = datetime.now(), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="moodboards")
    moodboard_canvases = relationship("MoodBoardCanvas", back_populates="moodboard", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<MoodBoard(id={self.id}, brand_name='{self.brand_name}', user_id={self.user_id})>"


class MoodBoardCanvas(Base):
    """MoodBoardCanvas model for storing generated canvas designs from moodboard"""
    
    __tablename__ = "moodboard_canvas"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    moodboard_id = Column(UUID(as_uuid=True), ForeignKey("moodboard.id", ondelete="CASCADE"), nullable=False)
    
    # Canvas information
    name = Column(Text, nullable=False, default="Untitled Canvas")
    canvas_urls = Column(JSONB, nullable=False)  # Array of generated canvas URLs in same order as moodboard images
    prompt = Column(Text, nullable=False)  # The prompt used to generate these canvases
    
    # Additional metadata
    is_favorite = Column(Boolean, default=False, nullable=False)
    generation_params = Column(JSONB, nullable=True)  # Store any generation parameters used
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default = datetime.now(), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default = datetime.now(), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    moodboard = relationship("MoodBoard", back_populates="moodboard_canvases")
    
    def __repr__(self):
        return f"<MoodBoardCanvas(id={self.id}, name='{self.name}', moodboard_id={self.moodboard_id})>"