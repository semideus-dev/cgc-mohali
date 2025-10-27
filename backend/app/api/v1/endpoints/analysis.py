"""API Endpoints for Ad Analysis"""

import uuid
import logging
from typing import Annotated
import os
import pytesseract
from io import BytesIO
from torchvision import models, transforms
import torch
from PIL import Image
import httpx
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import AnalysisJob
from app.db.schemas import AnalysisJobCreateResponse, AnalysisJobStatusResponse
from app.services.storage import StorageService
from app.services.processing import run_full_analysis

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", status_code=202, response_model=AnalysisJobCreateResponse)
async def create_analysis(
    file: Annotated[UploadFile, File(description="Ad image to analyze")],
    background_tasks: BackgroundTasks,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Upload an ad image for analysis.
    
    This endpoint:
    1. Accepts an image file
    2. Uploads it to storage
    3. Creates an analysis job in the database
    4. Starts background processing
    5. Returns the job ID immediately (202 Accepted)
    
    The actual analysis happens asynchronously in the background.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read image bytes
        logger.info(f"Receiving image: {file.filename}, type: {file.content_type}")
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
            status="processing",
            original_image_url=image_url
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        
        logger.info(f"Created analysis job: {job.id}")
        
        # Add background task for full analysis
        # Pass the app.state object which contains all loaded ML models
        background_tasks.add_task(
            run_full_analysis,
            job_id=str(job.id),
            image_bytes=image_bytes,
            app_state=request.app.state
        )
        
        logger.info(f"Background analysis task queued for job {job.id}")
        
        return AnalysisJobCreateResponse(job_id=job.id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating analysis job: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create analysis job: {str(e)}"
        )


@router.get("/results/{job_id}", response_model=AnalysisJobStatusResponse)
async def get_analysis_results(
    job_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Get the status and results of an analysis job.
    
    Args:
        job_id: UUID of the analysis job
        
    Returns:
        Job status and results (if completed)
        
    Status values:
    - "processing": Analysis is still running
    - "completed": Analysis finished successfully, results available
    - "failed": Analysis failed, error details in results
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