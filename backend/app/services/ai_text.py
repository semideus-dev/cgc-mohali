"""Text Analysis Services: OCR and Sentiment Analysis"""

import io
import logging
from typing import Dict

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def extract_text(ocr_reader, image_bytes: bytes) -> str:
    """
    Extract text from image using EasyOCR.
    
    Args:
        ocr_reader: The loaded EasyOCR Reader instance
        image_bytes: Raw image bytes
        
    Returns:
        Extracted text as a single string
    """
    try:
        # Convert bytes to numpy array for EasyOCR
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)
        
        # Run OCR
        results = ocr_reader.readtext(image_array, paragraph=True)
        
        # Extract text from results
        # EasyOCR returns list of tuples: (bbox, text, confidence)
        extracted_texts = []
        for detection in results:
            if len(detection) >= 2:
                text = detection[1]
                extracted_texts.append(text)
        
        # Join all text with spaces
        full_text = " ".join(extracted_texts)
        
        logger.info(f"Extracted {len(extracted_texts)} text segments, total length: {len(full_text)}")
        return full_text.strip()
        
    except Exception as e:
        logger.error(f"Error in text extraction: {e}")
        return ""


def analyze_sentiment(text_pipeline, text: str) -> Dict:
    """
    Analyze sentiment of the extracted text using DistilBERT.
    
    Args:
        text_pipeline: The loaded transformers sentiment analysis pipeline
        text: Text to analyze
        
    Returns:
        Dictionary containing sentiment label and score
    """
    try:
        if not text or len(text.strip()) == 0:
            return {
                "label": "NEUTRAL",
                "score": 0.0,
                "message": "No text to analyze"
            }
        
        # Truncate text if too long (BERT models have max token limits)
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")
        
        # Run sentiment analysis
        results = text_pipeline(text)
        
        # Extract result (pipeline returns a list with one dict)
        if results and len(results) > 0:
            result = results[0]
            label = result.get("label", "UNKNOWN")
            score = result.get("score", 0.0)
            
            logger.info(f"Sentiment analysis: {label} ({score:.3f})")
            return {
                "label": label,
                "score": round(float(score), 4)
            }
        else:
            return {
                "label": "UNKNOWN",
                "score": 0.0,
                "message": "No sentiment detected"
            }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {
            "label": "ERROR",
            "score": 0.0,
            "error": str(e)
        }

