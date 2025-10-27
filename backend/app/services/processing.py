"""Main Background Processing Service - Orchestrates all ML analysis"""

import logging
from typing import Any

from app.db.database import SessionLocal
from app.db.models import AnalysisJob
from app.services.ai_vision import analyze_objects, analyze_colors
from app.services.ai_text import extract_text, analyze_sentiment
from app.services.ai_generator import generate_image_with_diffusers
from app.services.storage import StorageService

logger = logging.getLogger(__name__)


def run_full_analysis(job_id: str, image_bytes: bytes, app_state: Any):
    """
    Main background task that runs complete ad analysis and generates new image.
    
    This function:
    1. Extracts all ML models from app_state
    2. Runs object detection, color analysis, OCR, and sentiment analysis
    3. Generates a critique of the ad
    4. Creates a prompt and generates a new image
    5. Uploads the new image to storage
    6. Updates the database with results
    
    Args:
        job_id: UUID of the analysis job
        image_bytes: Original image bytes
        app_state: FastAPI app.state object containing loaded ML models
    """
    # Create a new database session for this background task
    db = SessionLocal()
    
    try:
        logger.info(f"Starting analysis for job {job_id}")
        
        # Extract all models from app_state
        vision_model = app_state.vision_model
        coco_classes = app_state.vision_COCO_CLASSES
        device = app_state.device
        ocr_reader = app_state.ocr_reader
        text_pipeline = app_state.text_pipeline
        diffusers_pipeline = app_state.diffusers_pipeline
        
        # Step 1: Object Detection
        logger.info("Running object detection...")
        objects_result = analyze_objects(vision_model, coco_classes, device, image_bytes)
        
        # Step 2: Color Analysis
        logger.info("Running color analysis...")
        colors_result = analyze_colors(image_bytes, num_colors=5)
        
        # Step 3: Text Extraction (OCR)
        logger.info("Extracting text from image...")
        extracted_text = extract_text(ocr_reader, image_bytes)
        
        # Step 4: Sentiment Analysis (if text was found)
        sentiment_result = None
        if extracted_text and len(extracted_text.strip()) > 0:
            logger.info("Analyzing text sentiment...")
            sentiment_result = analyze_sentiment(text_pipeline, extracted_text)
        else:
            logger.info("No text found in image, skipping sentiment analysis")
            sentiment_result = {
                "label": "NEUTRAL",
                "score": 0.0,
                "message": "No text detected"
            }
        
        # Step 5: Synthesize a critique
        critique = synthesize_critique(
            objects_result,
            colors_result,
            extracted_text,
            sentiment_result
        )
        
        # Step 6: Generate a prompt for image generation
        generation_prompt = create_generation_prompt(
            objects_result,
            colors_result,
            extracted_text,
            critique
        )
        
        # Step 7: Generate new image using Stable Diffusion
        logger.info("Generating enhanced image with Stable Diffusion...")
        generated_image_bytes = generate_image_with_diffusers(
            pipeline=diffusers_pipeline,
            device=device,
            prompt=generation_prompt,
            image_bytes=image_bytes,
            num_inference_steps=20,  # Low for speed on CPU
            strength=0.6  # Moderate transformation
        )
        
        # Step 8: Upload generated image to storage
        logger.info("Uploading generated image to storage...")
        storage_service = StorageService()
        new_image_url = storage_service.upload_file(
            file_bytes=generated_image_bytes,
            file_name=f"generated_{job_id}.png",
            content_type="image/png"
        )
        
        # Step 9: Compile final results
        final_results = {
            "analysis": {
                "objects": objects_result,
                "colors": colors_result,
                "text": {
                    "extracted": extracted_text,
                    "sentiment": sentiment_result
                }
            },
            "critique": critique,
            "generated_image_url": new_image_url,
            "generation_prompt": generation_prompt
        }
        
        # Step 10: Update database with results
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.status = "completed"
            job.results = final_results
            db.commit()
            logger.info(f"Analysis completed successfully for job {job_id}")
        else:
            logger.error(f"Job {job_id} not found in database")
            
    except Exception as e:
        logger.error(f"Error during analysis for job {job_id}: {e}", exc_info=True)
        
        # Update job status to failed
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.results = {
                    "error": str(e),
                    "message": "Analysis failed due to an error"
                }
                db.commit()
        except Exception as db_error:
            logger.error(f"Failed to update job status: {db_error}")
    
    finally:
        db.close()


def synthesize_critique(objects_result: dict, colors_result: dict, text: str, sentiment: dict) -> dict:
    """
    Synthesize a comprehensive critique of the advertisement.
    
    Args:
        objects_result: Object detection results
        colors_result: Color analysis results
        text: Extracted text
        sentiment: Sentiment analysis results
        
    Returns:
        Dictionary containing structured critique
    """
    critique = {
        "visual_elements": [],
        "color_scheme": [],
        "messaging": [],
        "overall_score": 0.0
    }
    
    # Analyze visual elements
    objects = objects_result.get("objects", {})
    if len(objects) > 0:
        critique["visual_elements"].append(f"Detected {len(objects)} types of objects")
        for obj, data in list(objects.items())[:3]:  # Top 3
            critique["visual_elements"].append(
                f"{obj}: {data['count']} instance(s) with {data['max_confidence']:.0%} confidence"
            )
    else:
        critique["visual_elements"].append("No prominent objects detected")
    
    # Analyze color scheme
    dominant_colors = colors_result.get("dominant_colors", [])
    if dominant_colors:
        primary = dominant_colors[0]
        critique["color_scheme"].append(
            f"Primary color: {primary['hex']} ({primary['percentage']:.1f}% of image)"
        )
        critique["color_scheme"].append(
            f"Uses {len(dominant_colors)} distinct colors"
        )
    
    # Analyze messaging
    if text and len(text.strip()) > 0:
        critique["messaging"].append(f"Contains {len(text.split())} words")
        if sentiment:
            label = sentiment.get("label", "UNKNOWN")
            score = sentiment.get("score", 0)
            critique["messaging"].append(
                f"Sentiment: {label} (confidence: {score:.0%})"
            )
    else:
        critique["messaging"].append("No text detected in advertisement")
    
    # Calculate overall score (simple heuristic)
    score = 50.0  # Base score
    if len(objects) > 0:
        score += 15
    if dominant_colors:
        score += 10
    if text and len(text.strip()) > 0:
        score += 15
    if sentiment and sentiment.get("label") == "POSITIVE":
        score += 10
    
    critique["overall_score"] = min(score, 100.0)
    
    return critique


def create_generation_prompt(
    objects_result: dict,
    colors_result: dict,
    text: str,
    critique: dict
) -> str:
    """
    Create a text prompt for image generation based on analysis.
    
    Args:
        objects_result: Object detection results
        colors_result: Color analysis results
        text: Extracted text
        critique: Generated critique
        
    Returns:
        Text prompt for Stable Diffusion
    """
    prompt_parts = ["high quality professional advertisement,"]
    
    # Add objects
    objects = objects_result.get("objects", {})
    if objects:
        obj_names = list(objects.keys())[:3]
        prompt_parts.append(f"featuring {', '.join(obj_names)},")
    
    # Add color theme
    dominant_colors = colors_result.get("dominant_colors", [])
    if dominant_colors:
        primary_color = dominant_colors[0].get("hex", "")
        # Convert hex to color name (simplified)
        prompt_parts.append(f"vibrant colors,")
    
    # Add style
    prompt_parts.append("modern design, clean composition, professional photography,")
    prompt_parts.append("sharp focus, detailed, 8k resolution")
    
    # Join all parts
    prompt = " ".join(prompt_parts)
    
    # Limit prompt length
    if len(prompt) > 500:
        prompt = prompt[:500]
    
    return prompt

