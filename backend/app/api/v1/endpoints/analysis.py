"""Split API Endpoints for Ad Analysis Pipeline"""

import uuid
import logging
import asyncio
from typing import Annotated, Optional

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Request, Form
from sqlalchemy.orm import Session

import os
import pytesseract
from io import BytesIO
from torchvision import models, transforms
import torch
from PIL import Image
import httpx

from app.db.database import get_db
from app.db.models import AnalysisJob
from app.db.schemas import (
    UploadResponse, AnalysisResponse, CritiqueResponse, 
    GenerationResponse, AnalysisJobStatusResponse
)
from app.services.storage import StorageService
from app.services.ai_text import extract_text, analyze_comprehensive_text
from app.services.ai_vision import analyze_objects, analyze_colors
from app.services.ai_critique import generate_comprehensive_critique, generate_master_prompt
from app.services.ai_generator import generate_image_with_diffusers

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", status_code=201, response_model=UploadResponse)
async def upload_image_and_prompt(
    file: Annotated[UploadFile, File(description="Ad image to analyze")],
    prompt: Annotated[Optional[str], Form()] = None,
    db: Session = Depends(get_db)
):
    """
    Step 1: Upload image and optional user prompt.
    
    Creates analysis job and uploads image to storage.
    Returns job_id for subsequent pipeline steps.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read image bytes
        logger.info(f"Uploading image: {file.filename}, type: {file.content_type}")
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Upload original image to storage
        storage_service = StorageService()
        original_filename = file.filename or "uploaded_image.jpg"
        image_url = storage_service.upload_file(
            file_bytes=image_bytes,
            file_name=f"original_{original_filename}",
            content_type=file.content_type
        )
        
        logger.info(f"Uploaded original image to: {image_url}")
        
        # Create analysis job in database
        job = AnalysisJob(
            id=uuid.uuid4(),
            status="uploaded",
            original_image_url=image_url,
            user_prompt=prompt
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        
        logger.info(f"Created analysis job: {job.id}")
        
        return UploadResponse(
            job_id=job.id,
            status="uploaded",
            original_image_url=image_url,
            message="Image uploaded successfully. Proceed to /analyze endpoint."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload image: {str(e)}"
        )


@router.post("/analyze/{job_id}", status_code=200, response_model=AnalysisResponse)
async def analyze_image(
    job_id: uuid.UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Step 2: Run OCR, NLP, and Vision analysis concurrently.
    
    Performs comprehensive analysis of the uploaded image:
    - OCR text extraction
    - Text sentiment and NLP analysis  
    - Object detection and color analysis
    
    Runs asynchronously and updates job with results.
    """
    try:
        # Get job from database
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        
        if job.status != "uploaded":
            raise HTTPException(
                status_code=400, 
                detail=f"Job status is '{job.status}', expected 'uploaded'"
            )
        
        # Update status
        job.status = "analyzing"
        db.commit()
        
        # Get image bytes from storage URL (or cache them during upload)
        # For now, we'll need to re-download or pass them differently
        # This is a limitation of the current architecture
        
        # Add background task for analysis
        background_tasks.add_task(
            run_analysis_pipeline,
            job_id=str(job_id),
            app_state=request.app.state,
            db_session_factory=db.get_bind
        )
        
        return AnalysisResponse(
            job_id=job_id,
            status="analyzing",
            ocr_results=None,
            text_analysis=None,
            image_analysis=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis for job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start analysis: {str(e)}"
        )


@router.post("/critique/{job_id}", status_code=200, response_model=CritiqueResponse)
async def generate_critique(
    job_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Step 3: Generate comprehensive critique and master prompt.
    
    Takes analysis results and user prompt to create:
    - Comprehensive advertisement critique
    - Master prompt for image generation
    """
    try:
        # Get job from database
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        
        if job.status != "analyzed":
            raise HTTPException(
                status_code=400,
                detail=f"Job status is '{job.status}', expected 'analyzed'"
            )
        
        # Check if analysis results exist
        if not job.ocr_results or not job.text_analysis or not job.image_analysis:
            raise HTTPException(
                status_code=400,
                detail="Analysis results not available. Run /analyze first."
            )
        
        # Generate critique
        critique = generate_comprehensive_critique(
            ocr_results=job.ocr_results,
            text_analysis=job.text_analysis,
            image_analysis=job.image_analysis,
            user_prompt=job.user_prompt
        )
        
        # Generate master prompt
        master_prompt = generate_master_prompt(
            ocr_results=job.ocr_results,
            text_analysis=job.text_analysis,
            image_analysis=job.image_analysis,
            critique=critique,
            user_prompt=job.user_prompt
        )
        
        # Update job with critique and master prompt
        job.critique = critique
        job.master_prompt = master_prompt
        job.status = "critiqued"
        db.commit()
        
        logger.info(f"Generated critique and master prompt for job {job_id}")
        
        return CritiqueResponse(
            job_id=job_id,
            status="critiqued",
            critique=critique,
            master_prompt=master_prompt
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating critique for job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate critique: {str(e)}"
        )


@router.post("/generate/{job_id}", status_code=200, response_model=GenerationResponse)
async def generate_enhanced_image(
    job_id: uuid.UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Step 4: Generate enhanced image using master prompt.
    
    Uses Stable Diffusion with the master prompt to create
    an enhanced version of the original advertisement.
    """
    try:
        # Get job from database
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        
        if job.status != "critiqued":
            raise HTTPException(
                status_code=400,
                detail=f"Job status is '{job.status}', expected 'critiqued'"
            )
        
        if not job.master_prompt:
            raise HTTPException(
                status_code=400,
                detail="Master prompt not available. Run /critique first."
            )
        
        # Update status
        job.status = "generating"
        db.commit()
        
        # Add background task for image generation
        background_tasks.add_task(
            run_generation_pipeline,
            job_id=str(job_id),
            app_state=request.app.state,
            db_session_factory=db.get_bind
        )
        
        return GenerationResponse(
            job_id=job_id,
            status="generating",
            generated_image_url=None,
            master_prompt=job.master_prompt
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting generation for job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start generation: {str(e)}"
        )


@router.get("/status/{job_id}", response_model=AnalysisJobStatusResponse)
async def get_job_status(
    job_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive status and results of an analysis job.
    
    Returns complete job information including all analysis results,
    critique, and generated images if available.
    """
    try:
        # Query the job from database
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis job {job_id} not found"
            )
        
        logger.info(f"Retrieved job {job_id}, status: {job.status}")
        
        return AnalysisJobStatusResponse.model_validate(job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve job: {str(e)}"
        )


@router.post("/generate-image")
async def generate_image(prompt : str):
    generated_ad = httpx.get(f'https://image.pollinations.ai/prompt/{prompt}')
    res = httpx.post(
        "https://uploadthing.com/api/uploadFiles",
        headers={
            "x-uploadthing-api-key": os.getenv("UPLOADTHING_SECRET"),
            "Content-Type": "application/json"
        },
        json={
            "files": [
                {
                    "name": "generated.png",
                    "size": len(generated_ad.content),
                    "type": "image/png"
                }
            ],
            "appId": os.getenv("UPLOADTHING_APP_ID")
        }
    )
    upload_info = res.json()["data"][0]
    url = upload_info["url"]
    fields = upload_info["fields"]
    file_url = upload_info["fileUrl"]
    multipart_data = {**fields, "file": ("generated.png", generated_ad.content, "image/png")}
    httpx.post(url, files=multipart_data)

    return {"imageUrl" : file_url}


@router.post("/run-ocr")
async def run_ocr(image_url: str):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image_bytes = httpx.get(image_url).content
    image = Image.open(BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    text = text.replace("httpsi//", "https://")
    text = text.replace("httpi//", "http://")

    return {"ocr_text": text.replace('\n', ' ')}

@router.post("/torch-analysis")
async def torch_analysis(image_url: str):
    model = models.resnet50(pretrained=True)
    model.eval()
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
    resp = httpx.post(image_url)
    image = Image.open(BytesIO(resp.content)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get top prediction
    top_prob, top_idx = torch.topk(probs, 1)
    from torchvision.models import ResNet50_Weights
    labels = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

    return {"label": labels[top_idx.item()], "confidence": float(top_prob.item()), "probs" : probs}

# Background task functions

async def run_analysis_pipeline(job_id: str, app_state, db_session_factory):
    """Background task to run OCR, NLP, and Vision analysis concurrently"""
    from app.db.database import SessionLocal
    
    db = SessionLocal()
    try:
        logger.info(f"Starting analysis pipeline for job {job_id}")
        
        # Get job
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        # We need to get the image bytes - this is a limitation of the split architecture
        # In production, you'd want to cache the image bytes or use a different approach
        # For now, we'll need to re-download from the storage URL
        
        # Get models from app_state
        ocr_reader = app_state.ocr_reader
        text_pipeline = app_state.text_pipeline
        vision_model = app_state.vision_model
        coco_classes = app_state.vision_COCO_CLASSES
        device = app_state.device
        
        # Download image bytes from storage URL
        storage_service = StorageService()
        try:
            image_bytes = storage_service.download_file(job.original_image_url)
        except Exception as e:
            logger.error(f"Could not download image for job {job_id}: {e}")
            job.status = "failed"
            db.commit()
            return
        
        # Run analysis tasks concurrently
        ocr_task = asyncio.create_task(
            asyncio.to_thread(extract_text, ocr_reader, image_bytes)
        )
        
        vision_objects_task = asyncio.create_task(
            asyncio.to_thread(analyze_objects, vision_model, coco_classes, device, image_bytes)
        )
        
        vision_colors_task = asyncio.create_task(
            asyncio.to_thread(analyze_colors, image_bytes)
        )
        
        # Wait for all tasks to complete
        ocr_results, objects_results, colors_results = await asyncio.gather(
            ocr_task, vision_objects_task, vision_colors_task
        )
        
        # Run text analysis if OCR found text
        text_analysis = {}
        if ocr_results.get("raw_text"):
            text_analysis = analyze_comprehensive_text(
                text_pipeline, 
                ocr_results["raw_text"]
            )
        
        # Combine image analysis results
        image_analysis = {
            "objects": objects_results,
            "colors": colors_results
        }
        
        # Update job with results
        job.ocr_results = ocr_results
        job.text_analysis = text_analysis
        job.image_analysis = image_analysis
        job.status = "analyzed"
        db.commit()
        
        logger.info(f"Analysis pipeline completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline for job {job_id}: {e}", exc_info=True)
        job.status = "failed"
        db.commit()
    finally:
        db.close()


async def run_generation_pipeline(job_id: str, app_state, db_session_factory):
    """Background task to generate enhanced image"""
    from app.db.database import SessionLocal
    
    db = SessionLocal()
    try:
        logger.info(f"Starting generation pipeline for job {job_id}")
        
        # Get job
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        # Get models from app_state
        diffusers_pipeline = app_state.diffusers_pipeline
        device = app_state.device
        
        # Download original image bytes from storage URL
        storage_service = StorageService()
        try:
            image_bytes = storage_service.download_file(job.original_image_url)
        except Exception as e:
            logger.error(f"Could not download image for job {job_id}: {e}")
            job.status = "failed"
            db.commit()
            return
        
        # Generate enhanced image
        generated_image_bytes = generate_image_with_diffusers(
            pipeline=diffusers_pipeline,
            device=device,
            prompt=job.master_prompt,
            image_bytes=image_bytes
        )
        
        # Upload generated image
        storage_service = StorageService()
        generated_image_url = storage_service.upload_file(
            file_bytes=generated_image_bytes,
            file_name=f"generated_{job_id}.png",
            content_type="image/png"
        )
        
        # Update job with generated image
        job.generated_image_url = generated_image_url
        job.status = "completed"
        db.commit()
        
        logger.info(f"Generation pipeline completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Error in generation pipeline for job {job_id}: {e}", exc_info=True)
        job.status = "failed"
        db.commit()
    finally:
        db.close()

