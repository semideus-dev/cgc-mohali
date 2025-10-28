"""API Endpoints for Ad Analysis"""

import uuid
import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import AnalysisJob, MoodBoard, MoodBoardCanvas
from app.db.schemas import (
    AnalysisJobCreateResponse, 
    AnalysisJobStatusResponse, 
    GenerationResponse,
    MoodBoardCreate,
    MoodBoardUpdate,
    MoodBoardResponse,
    MoodBoardCanvasResponse
)
from app.services.storage import StorageService
from app.core.config import settings
from app.services.ai_text import extract_text, analyze_comprehensive_text, generate_nlp_prompt
from app.services.ai_vision import analyze_objects, analyze_colors, generate_vision_prompt
from app.services.ai_critique import generate_comprehensive_critique, generate_master_prompt

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", status_code=201)
async def upload_image_and_prompt(
    file: Annotated[UploadFile, File(description="Ad image to analyze")],
    prompt: Annotated[str | None, Form()] = None,
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
        
        return {
            "job_id": job.id,
            "status": "uploaded",
            "original_image_url": image_url,
            "message": "Image uploaded successfully. Proceed to /analyze/{job_id} endpoint."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload image: {str(e)}"
        )


@router.post("/analyze/{job_id}", status_code=200)
async def analyze_image(
    job_id: uuid.UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Step 2: Run OCR, NLP, and Vision analysis concurrently.
    
    Performs comprehensive analysis and generates NLP + Vision prompts.
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
        
        logger.info(f"Starting analysis for job {job_id}")
        
        # Get image from storage
        storage_service = StorageService()
        image_bytes = storage_service.download_file(job.original_image_url)
        
        # Get ML models from app state
        ocr_reader = request.app.state.ocr_reader
        text_pipeline = request.app.state.text_pipeline
        vision_model = request.app.state.vision_model
        coco_classes = request.app.state.vision_COCO_CLASSES
        device = request.app.state.device
        
        # Run OCR
        logger.info("Running OCR...")
        ocr_results = extract_text(ocr_reader, image_bytes)
        
        # Run NLP analysis if text found
        text_analysis = {}
        if ocr_results.get("raw_text"):
            logger.info("Running NLP analysis...")
            text_analysis = analyze_comprehensive_text(text_pipeline, ocr_results["raw_text"])
        
        # Run vision analysis
        logger.info("Running vision analysis...")
        objects_results = analyze_objects(vision_model, coco_classes, device, image_bytes)
        colors_results = analyze_colors(image_bytes)
        
        image_analysis = {
            "objects": objects_results,
            "colors": colors_results
        }
        
        # Generate specialized prompts
        logger.info("Generating NLP and Vision prompts...")
        nlp_prompt = generate_nlp_prompt(ocr_results, text_analysis)
        vision_prompt = generate_vision_prompt(image_analysis)
        
        # Update job with results
        job.ocr_results = ocr_results
        job.text_analysis = text_analysis
        job.image_analysis = image_analysis
        job.status = "analyzed"
        db.commit()
        
        logger.info(f"Analysis completed for job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "analyzed",
            "message": "Analysis completed. Proceed to /critique/{job_id} endpoint.",
            "nlp_prompt": nlp_prompt,
            "vision_prompt": vision_prompt
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing image for job {job_id}: {e}", exc_info=True)
        
        # Update job status to failed
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                job.status = "failed"
                db.commit()
        except Exception as db_error:
            logger.error(f"Failed to update job status: {db_error}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/critique/{job_id}", status_code=200)
async def generate_critique_and_master_prompt(
    job_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Step 3: Generate comprehensive critique and master prompt.
    
    Combines user prompt, NLP prompt, and vision prompt into final master prompt.
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
        
        logger.info(f"Generating critique and master prompt for job {job_id}")
        
        # Generate comprehensive critique
        critique = generate_comprehensive_critique(
            ocr_results=job.ocr_results,
            text_analysis=job.text_analysis,
            image_analysis=job.image_analysis,
            user_prompt=job.user_prompt
        )
        
        # Generate specialized prompts
        nlp_prompt = generate_nlp_prompt(job.ocr_results, job.text_analysis)
        vision_prompt = generate_vision_prompt(job.image_analysis)
        
        # Generate master prompt by combining all three prompts
        master_prompt = generate_master_prompt(
            nlp_prompt=nlp_prompt,
            vision_prompt=vision_prompt,
            critique=critique,
            user_prompt=job.user_prompt
        )
        
        # Update job with critique and master prompt
        job.critique = critique
        job.master_prompt = master_prompt
        job.status = "critiqued"
        db.commit()
        
        logger.info(f"Critique and master prompt generated for job {job_id}")
        
        return {
            "job_id": job_id,
            "status": "critiqued",
            "message": "Critique completed. Proceed to /generate/{job_id} endpoint.",
            "critique": critique,
            "master_prompt": master_prompt,
            "nlp_prompt": nlp_prompt,
            "vision_prompt": vision_prompt,
            "user_prompt": job.user_prompt
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating critique for job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate critique: {str(e)}"
        )


# @router.post("/generate/{job_id}", status_code=200, response_model=GenerationResponse)
# async def generate_enhanced_image(
#     job_id: uuid.UUID,
#     request: Request,
#     background_tasks: BackgroundTasks,
#     db: Session = Depends(get_db)
# ):
#     """
#     Step 4: Generate enhanced image using master prompt.
    
#     Uses Stable Diffusion with the master prompt to create
#     an enhanced version of the original advertisement.
#     """
#     try:
#         # Get job from database
#         job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
#         if not job:
#             raise HTTPException(status_code=404, detail="Analysis job not found")
        
#         if job.status != "critiqued":
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Job status is '{job.status}', expected 'critiqued'"
#             )
        
#         if not job.master_prompt:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Master prompt not available. Run /critique first."
#             )
        
#         # Update status
#         job.status = "generating"
#         db.commit()
        
#         # Add background task for image generation
#         background_tasks.add_task(
#             run_generation_pipeline,
#             job_id=str(job_id),
#             app_state=request.app.state,
#             db_session_factory=db.get_bind
#         )
        
#         return GenerationResponse(
#             job_id=job_id,
#             status="generating",
#             generated_image_url=None,
#             master_prompt=job.master_prompt
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error starting generation for job {job_id}: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to start generation: {str(e)}"
#         )


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


@router.post("/generate-image/{job_id}", status_code=200, response_model=GenerationResponse)
async def generate_image(
    job_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Step 4: Generate image using pollinations.ai API.
    
    Uses pollinations.ai API with the master prompt to create
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
        
        # Update status to generating
        job.status = "generating"
        db.commit()
        
        logger.info(f"Starting image generation for job {job_id} with prompt: {job.master_prompt}")
        
        # Generate image using pollinations.ai (async)
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Generate image from pollinations.ai
            generated_response = await client.get(
                f'https://image.pollinations.ai/prompt/{job.master_prompt}'
            )
            
            if generated_response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate image from pollinations.ai: {generated_response.status_code}"
                )
            
            generated_image_bytes = generated_response.content
            logger.info(f"Generated image size: {len(generated_image_bytes)} bytes")
            
            # Upload generated image to UploadThing
            # Step 1: Get presigned URL
            presigned_response = await client.post(
                "https://uploadthing.com/api/uploadFiles",
                headers={
                    "x-uploadthing-api-key": settings.UPLOADTHING_SECRET,
                    "Content-Type": "application/json"
                },
                json={
                    "files": [
                        {
                            "name": f"generated_{job_id}.png",
                            "size": len(generated_image_bytes),
                            "type": "image/png"
                        }
                    ],
                    "appId": settings.UPLOADTHING_APP_ID
                }
            )
            
            if presigned_response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get presigned URL: {presigned_response.status_code}"
                )
            
            upload_info = presigned_response.json()["data"][0]
            presigned_url = upload_info["url"]
            fields = upload_info["fields"]
            file_url = upload_info["fileUrl"]
            
            # Step 2: Upload to presigned URL
            multipart_data = {
                **fields,
                "file": (f"generated_{job_id}.png", generated_image_bytes, "image/png")
            }
            
            upload_response = await client.post(presigned_url, files=multipart_data)
            
            if upload_response.status_code not in [200, 201, 204]:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload to UploadThing: {upload_response.status_code}"
                )
            
            logger.info(f"Successfully uploaded generated image to: {file_url}")
        
        # Update job with generated image
        job.generated_image_url = file_url
        job.status = "completed"
        db.commit()
        
        logger.info(f"Image generation completed for job {job_id}")
        
        return GenerationResponse(
            job_id=job_id,
            status="completed",
            generated_image_url=file_url,
            master_prompt=job.master_prompt
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating image for job {job_id}: {e}", exc_info=True)
        
        # Update job status to failed
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                job.status = "failed"
                db.commit()
        except Exception as db_error:
            logger.error(f"Failed to update job status: {db_error}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )

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
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    resp = httpx.get(image_url)
    image = Image.open(BytesIO(resp.content)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get top prediction
    top_prob, top_idx = torch.topk(probs, 1)
    labels = weights.meta["categories"]
    labels = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    return {
        "label": labels[top_idx.item()],
        "confidence": float(top_prob.item()),
        "top5": [
            {"label": labels[idx], "prob": float(prob)}
            for prob, idx in zip(*torch.topk(probs, 5))
        ]
    }

@router.post("/critique-analyser")
async def critique_analyser(ocr : str, torch_analysis: str, canvas_id : str, db: Session = Depends(get_db)):
    canvas_prompt = db.query(Canvas).filter(Canvas.id == canvas_id).first().prompt
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    headers = {
    "x-goog-api-key": "AIzaSyDh3HjAgjV7Ij7_nqIm3pzT_SI0I2iUUp4",
    "Content-Type": "application/json"
}

    data = {
    "contents": [
        {
            "parts": [
                {"text": """ You are a creative ad banner designer and visual art prompt engineer.

You will receive:
1. An OCR text extraction report — text found in the input image.
2. A TorchVision analysis report — objects, scene, and entities detected in the image.

Your goal:
- Analyze both reports together.
- Identify the product, service, or theme the ad likely represents.
- Then, rewrite and expand it into a **highly descriptive prompt** for an image generation model.
- The goal is to generate a new, professional, visually engaging ad banner that clearly conveys the theme.
- The prompt should guide the model to produce an image that looks realistic, balanced, and visually appealing.

Guidelines:
- The ad should not be vague, minimal, or empty — it should include **composition details**, **lighting**, **color palette**, **objects**, **background style**, and **context**.
- Always mention **key visual elements** like text placement, logo positioning, and the general aesthetic (modern, clean, luxury, youthful, techy, etc.).
- Avoid random artistic vagueness. Focus on **commercial, marketing-ready visuals**.
- Preserve the meaning of the original ad while enhancing clarity, attractiveness, and professionalism.

Example input:
OCR: "Summer Sale 50% Off on Sunglasses"
Torch Analysis: "Detected objects: sunglasses, woman face, beach background"


"refined_prompt": "a vibrant summer sale ad banner showing a woman wearing stylish sunglasses at the beach, tropical background with palm trees and bright sunlight, bold text 'Summer Sale 50% Off' in modern sans-serif font, brand logo in the corner, realistic lighting, glossy finish, fashion advertisement aesthetic"


Now generate the best possible descriptive ad banner prompt using the following input:
 """ + " here is the user's expectations: " + canvas_prompt + " here is the ocr report: " + ocr + " and here is the torchvision report" + torch_analysis}
            ]
        }
    ]
}

    response = httpx.post(url, headers=headers, json=data, timeout=httpx.Timeout(60.0, read=60.0))

    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


# ============= MoodBoard Endpoints =============

@router.post("/moodboard/create", status_code=201, response_model=MoodBoardResponse)
async def create_moodboard(
    moodboard_data: MoodBoardCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new moodboard with brand information, images, and prompt.
    
    This endpoint creates a moodboard that can contain up to 5+ images
    with a single prompt describing the brand vision.
    """
    try:
        # Create moodboard
        moodboard = MoodBoard(
            id=uuid.uuid4(),
            user_id=moodboard_data.user_id,
            brand_name=moodboard_data.brand_name,
            brand_slogan=moodboard_data.brand_slogan,
            description=moodboard_data.description,
            color_palette=moodboard_data.color_palette,
            images=moodboard_data.images,
            prompt=moodboard_data.prompt
        )
        
        db.add(moodboard)
        db.commit()
        db.refresh(moodboard)
        
        logger.info(f"Created moodboard: {moodboard.id} with {len(moodboard_data.images or [])} images")
        
        return MoodBoardResponse.model_validate(moodboard)
        
    except Exception as e:
        logger.error(f"Error creating moodboard: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create moodboard: {str(e)}"
        )


@router.get("/moodboard/{moodboard_id}", response_model=MoodBoardResponse)
async def get_moodboard(
    moodboard_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get a specific moodboard by ID."""
    try:
        moodboard = db.query(MoodBoard).filter(MoodBoard.id == moodboard_id).first()
        
        if not moodboard:
            raise HTTPException(status_code=404, detail="Moodboard not found")
        
        return MoodBoardResponse.model_validate(moodboard)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving moodboard {moodboard_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve moodboard: {str(e)}"
        )


@router.post("/moodboard/{moodboard_id}/analyze", status_code=202)
async def analyze_moodboard(
    moodboard_id: uuid.UUID,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze all images in a moodboard in parallel.
    
    This endpoint:
    1. Retrieves all images from the moodboard
    2. Runs OCR, NLP, and Vision analysis on each image in parallel
    3. Combines all analyses with the brand color palette
    4. Generates a comprehensive master prompt for each image
    5. Returns a processing status
    
    Images are processed in parallel but results maintain order.
    """
    try:
        # Get moodboard
        moodboard = db.query(MoodBoard).filter(MoodBoard.id == moodboard_id).first()
        if not moodboard:
            raise HTTPException(status_code=404, detail="Moodboard not found")
        
        if not moodboard.images or len(moodboard.images) == 0:
            raise HTTPException(status_code=400, detail="Moodboard has no images")
        
        logger.info(f"Starting analysis for moodboard {moodboard_id} with {len(moodboard.images)} images")
        
        # Add background task for parallel analysis
        background_tasks.add_task(
            analyze_moodboard_images_parallel,
            moodboard_id=str(moodboard_id),
            app_state=request.app.state,
            db_session_factory=db.get_bind
        )
        
        return {
            "moodboard_id": moodboard_id,
            "status": "analyzing",
            "image_count": len(moodboard.images),
            "message": f"Analysis started for {len(moodboard.images)} images. This will run in parallel."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting moodboard analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start analysis: {str(e)}"
        )


@router.post("/moodboard/{moodboard_id}/generate", status_code=200)
async def generate_moodboard_canvases(
    moodboard_id: uuid.UUID,
    canvas_name: str = "Generated Canvas",
    db: Session = Depends(get_db)
):
    """
    Generate enhanced canvas images for all moodboard images.
    
    This endpoint:
    1. Retrieves the moodboard with all analysis results
    2. For each image, generates an enhanced version using pollinations.ai
    3. Maintains the order of images
    4. Creates a MoodBoardCanvas with all generated URLs
    
    The generation happens in parallel but order is preserved.
    """
    try:
        # Get moodboard
        moodboard = db.query(MoodBoard).filter(MoodBoard.id == moodboard_id).first()
        if not moodboard:
            raise HTTPException(status_code=404, detail="Moodboard not found")
        
        if not moodboard.images or len(moodboard.images) == 0:
            raise HTTPException(status_code=400, detail="Moodboard has no images")
        
        logger.info(f"Starting image generation for moodboard {moodboard_id}")
        
        # Generate images in parallel while maintaining order
        canvas_urls = await generate_moodboard_images_parallel(
            moodboard=moodboard,
            db=db
        )
        
        # Create MoodBoardCanvas
        canvas = MoodBoardCanvas(
            id=uuid.uuid4(),
            moodboard_id=moodboard_id,
            name=canvas_name,
            canvas_urls=canvas_urls,
            prompt=moodboard.prompt or "Professional brand canvas",
            is_favorite=False,
            generation_params={
                "color_palette": moodboard.color_palette,
                "brand_name": moodboard.brand_name,
                "image_count": len(canvas_urls)
            }
        )
        
        db.add(canvas)
        db.commit()
        db.refresh(canvas)
        
        logger.info(f"Created canvas {canvas.id} with {len(canvas_urls)} generated images")
        
        return MoodBoardCanvasResponse.model_validate(canvas)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating moodboard canvases: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate canvases: {str(e)}"
        )


@router.get("/moodboard/{moodboard_id}/canvases")
async def get_moodboard_canvases(
    moodboard_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get all canvases for a specific moodboard."""
    try:
        canvases = db.query(MoodBoardCanvas).filter(
            MoodBoardCanvas.moodboard_id == moodboard_id
        ).all()
        
        return [MoodBoardCanvasResponse.model_validate(canvas) for canvas in canvases]
        
    except Exception as e:
        logger.error(f"Error retrieving canvases: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve canvases: {str(e)}"
        )


# ============= Helper Functions for Moodboard Processing =============

async def analyze_single_moodboard_image(
    image_url: str,
    image_index: int,
    moodboard_prompt: str,
    color_palette: list,
    brand_name: str,
    ocr_reader,
    text_pipeline,
    vision_model,
    coco_classes,
    device
) -> dict:
    """
    Analyze a single moodboard image and generate specialized prompts.
    
    Returns analysis results with index to maintain order.
    """
    try:
        logger.info(f"Analyzing moodboard image #{image_index}: {image_url[:50]}...")
        
        # Download image
        storage_service = StorageService()
        image_bytes = storage_service.download_file(image_url)
        
        # Run OCR
        ocr_results = extract_text(ocr_reader, image_bytes)
        
        # Run NLP analysis if text found
        text_analysis = {}
        if ocr_results.get("raw_text"):
            text_analysis = analyze_comprehensive_text(text_pipeline, ocr_results["raw_text"])
        
        # Run vision analysis
        objects_results = analyze_objects(vision_model, coco_classes, device, image_bytes)
        colors_results = analyze_colors(image_bytes)
        
        image_analysis = {
            "objects": objects_results,
            "colors": colors_results
        }
        
        # Generate specialized prompts
        nlp_prompt = generate_nlp_prompt(ocr_results, text_analysis)
        vision_prompt = generate_vision_prompt(image_analysis)
        
        # Generate critique with brand context
        critique = generate_comprehensive_critique(
            ocr_results=ocr_results,
            text_analysis=text_analysis,
            image_analysis=image_analysis,
            user_prompt=f"{brand_name} - {moodboard_prompt}"
        )
        
        # Build color palette description
        color_palette_desc = ""
        if color_palette and len(color_palette) > 0:
            color_palette_desc = f"using brand colors: {', '.join(color_palette)}"
        
        # Generate master prompt with brand context
        master_prompt = generate_master_prompt(
            nlp_prompt=nlp_prompt,
            vision_prompt=vision_prompt,
            critique=critique,
            user_prompt=f"{moodboard_prompt}. {color_palette_desc}. Brand: {brand_name}"
        )
        
        logger.info(f"Completed analysis for image #{image_index}")
        
        return {
            "index": image_index,
            "image_url": image_url,
            "ocr_results": ocr_results,
            "text_analysis": text_analysis,
            "image_analysis": image_analysis,
            "critique": critique,
            "nlp_prompt": nlp_prompt,
            "vision_prompt": vision_prompt,
            "master_prompt": master_prompt
        }
        
    except Exception as e:
        logger.error(f"Error analyzing moodboard image #{image_index}: {e}", exc_info=True)
        return {
            "index": image_index,
            "image_url": image_url,
            "error": str(e),
            "master_prompt": f"Professional {brand_name} advertisement with {moodboard_prompt}"
        }


async def analyze_moodboard_images_parallel(
    moodboard_id: str,
    app_state,
    db_session_factory
):
    """
    Background task to analyze all moodboard images in parallel.
    
    Maintains order of results even though processing is parallel.
    """
    from sqlalchemy.orm import sessionmaker
    import asyncio
    
    Session = sessionmaker(bind=db_session_factory)
    db = Session()
    
    try:
        # Get moodboard
        moodboard = db.query(MoodBoard).filter(MoodBoard.id == moodboard_id).first()
        if not moodboard:
            logger.error(f"Moodboard {moodboard_id} not found")
            return
        
        # Get ML models
        ocr_reader = app_state.ocr_reader
        text_pipeline = app_state.text_pipeline
        vision_model = app_state.vision_model
        coco_classes = app_state.vision_COCO_CLASSES
        device = app_state.device
        
        logger.info(f"Starting parallel analysis of {len(moodboard.images)} images")
        
        # Create tasks for all images
        tasks = []
        for idx, image_url in enumerate(moodboard.images):
            task = analyze_single_moodboard_image(
                image_url=image_url,
                image_index=idx,
                moodboard_prompt=moodboard.prompt or "",
                color_palette=moodboard.color_palette or [],
                brand_name=moodboard.brand_name,
                ocr_reader=ocr_reader,
                text_pipeline=text_pipeline,
                vision_model=vision_model,
                coco_classes=coco_classes,
                device=device
            )
            tasks.append(task)
        
        # Run all tasks in parallel
        results = await asyncio.gather(*tasks)
        
        # Sort results by index to maintain order
        results = sorted(results, key=lambda x: x["index"])
        
        # Store results (you could add a JSONB field to MoodBoard for analysis results)
        logger.info(f"Completed parallel analysis of {len(results)} images for moodboard {moodboard_id}")
        
        # Store results as metadata (optional - add a field to MoodBoard model if needed)
        # For now, just log completion
        
        db.commit()
        
    except Exception as e:
        logger.error(f"Error in parallel moodboard analysis: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


async def generate_moodboard_images_parallel(
    moodboard: MoodBoard,
    db: Session
) -> list:
    """
    Generate enhanced images for all moodboard images in parallel.
    
    Maintains order of generated images to match original images.
    """
    import asyncio
    
    try:
        logger.info(f"Starting parallel image generation for {len(moodboard.images)} images")
        
        # Build color palette description
        color_palette_desc = ""
        if moodboard.color_palette and len(moodboard.color_palette) > 0:
            color_palette_desc = f"using brand colors {', '.join(moodboard.color_palette)}"
        
        # Create base prompt
        base_prompt = f"{moodboard.brand_name}. {moodboard.prompt or 'Professional brand advertisement'}. {color_palette_desc}"
        
        # Create tasks for all images
        tasks = []
        for idx, image_url in enumerate(moodboard.images):
            # Download and analyze each image first to create unique prompt
            task = generate_single_moodboard_image(
                image_url=image_url,
                image_index=idx,
                base_prompt=base_prompt,
                moodboard=moodboard
            )
            tasks.append(task)
        
        # Run all tasks in parallel
        results = await asyncio.gather(*tasks)
        
        # Sort by index to maintain order
        results = sorted(results, key=lambda x: x["index"])
        
        # Extract URLs in order
        canvas_urls = [result["generated_url"] for result in results]
        
        logger.info(f"Generated {len(canvas_urls)} images in correct order")
        
        return canvas_urls
        
    except Exception as e:
        logger.error(f"Error in parallel image generation: {e}", exc_info=True)
        raise


async def generate_single_moodboard_image(
    image_url: str,
    image_index: int,
    base_prompt: str,
    moodboard: MoodBoard
) -> dict:
    """
    Generate a single enhanced image using pollinations.ai.
    
    Returns the generated URL with index to maintain order.
    """
    try:
        logger.info(f"Generating image #{image_index} from: {image_url[:50]}...")
        
        # Create unique prompt for this image
        prompt = f"{base_prompt}. Image {image_index + 1} of {len(moodboard.images)}"
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Generate image from pollinations.ai
            generated_response = await client.get(
                f'https://image.pollinations.ai/prompt/{prompt}'
            )
            
            if generated_response.status_code != 200:
                raise Exception(f"Pollinations.ai returned {generated_response.status_code}")
            
            generated_image_bytes = generated_response.content
            logger.info(f"Generated image #{image_index}: {len(generated_image_bytes)} bytes")
            
            # Upload to UploadThing
            presigned_response = await client.post(
                "https://uploadthing.com/api/uploadFiles",
                headers={
                    "x-uploadthing-api-key": settings.UPLOADTHING_SECRET,
                    "Content-Type": "application/json"
                },
                json={
                    "files": [{
                        "name": f"moodboard_{moodboard.id}_canvas_{image_index}.png",
                        "size": len(generated_image_bytes),
                        "type": "image/png"
                    }],
                    "appId": settings.UPLOADTHING_APP_ID
                }
            )
            
            if presigned_response.status_code != 200:
                raise Exception(f"UploadThing presigned URL failed: {presigned_response.status_code}")
            
            upload_info = presigned_response.json()["data"][0]
            presigned_url = upload_info["url"]
            fields = upload_info["fields"]
            file_url = upload_info["fileUrl"]
            
            # Upload to presigned URL
            multipart_data = {
                **fields,
                "file": (f"moodboard_canvas_{image_index}.png", generated_image_bytes, "image/png")
            }
            
            upload_response = await client.post(presigned_url, files=multipart_data)
            
            if upload_response.status_code not in [200, 201, 204]:
                raise Exception(f"Upload failed: {upload_response.status_code}")
            
            logger.info(f"Successfully generated and uploaded image #{image_index}")
            
            return {
                "index": image_index,
                "generated_url": file_url,
                "prompt": prompt
            }
            
    except Exception as e:
        logger.error(f"Error generating image #{image_index}: {e}", exc_info=True)
        # Return a placeholder or the original URL on error
        return {
            "index": image_index,
            "generated_url": image_url,  # Fallback to original
            "error": str(e)
        }