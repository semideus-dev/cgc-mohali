"""Enhanced Text Analysis Services: OCR, Sentiment, Emotion, and Comprehensive NLP"""

import io
import logging
import re
from typing import Dict, List
from collections import Counter

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def extract_text(ocr_reader, image_bytes: bytes) -> Dict:
    """
    Extract text from image using EasyOCR with detailed confidence metrics.
    
    Args:
        ocr_reader: The loaded EasyOCR Reader instance
        image_bytes: Raw image bytes
        
    Returns:
        Dictionary with extracted text and metadata
    """
    try:
        # Convert bytes to numpy array for EasyOCR
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)
        
        # Run OCR with detailed output
        results = ocr_reader.readtext(image_array, paragraph=False)
        
        # Process results
        extracted_texts = []
        confidences = []
        
        for detection in results:
            if len(detection) >= 3:
                bbox, text, confidence = detection[0], detection[1], detection[2]
                extracted_texts.append(text)
                confidences.append(confidence)
        
        # Join all text
        full_text = " ".join(extracted_texts)
        
        # Calculate metrics
        avg_confidence = np.mean(confidences) if confidences else 0.0
        word_count = len(full_text.split())
        char_count = len(full_text)
        
        logger.info(f"Extracted {len(extracted_texts)} text segments, avg confidence: {avg_confidence:.3f}")
        
        return {
            "raw_text": full_text.strip(),
            "segments": extracted_texts,
            "confidence": float(avg_confidence),
            "word_count": word_count,
            "char_count": char_count,
            "segment_count": len(extracted_texts)
        }
        
    except Exception as e:
        logger.error(f"Error in text extraction: {e}")
        return {
            "raw_text": "",
            "segments": [],
            "confidence": 0.0,
            "word_count": 0,
            "char_count": 0,
            "segment_count": 0,
            "error": str(e)
        }


def analyze_comprehensive_text(text_pipeline, text: str) -> Dict:
    """
    Comprehensive text analysis including sentiment, emotions, keywords, and more.
    
    Args:
        text_pipeline: The loaded transformers analysis pipeline
        text: Text to analyze
        
    Returns:
        Dictionary containing comprehensive text analysis
    """
    try:
        if not text or len(text.strip()) == 0:
            return {
                "sentiment": {"label": "NEUTRAL", "score": 0.0},
                "emotions": {},
                "keywords": [],
                "readability_score": 0.0,
                "tone": "neutral",
                "language": "unknown",
                "entities": [],
                "message": "No text to analyze"
            }
        
        # Sentiment Analysis
        sentiment = analyze_sentiment(text_pipeline, text)
        
        # Emotion Detection (simulated - would use emotion model in production)
        emotions = detect_emotions(text)
        
        # Keyword Extraction
        keywords = extract_keywords(text)
        
        # Readability Score
        readability = calculate_readability(text)
        
        # Tone Analysis
        tone = analyze_tone(text, sentiment)
        
        # Named Entity Recognition (basic pattern matching)
        entities = extract_entities(text)
        
        result = {
            "sentiment": sentiment,
            "emotions": emotions,
            "keywords": keywords,
            "readability_score": readability,
            "tone": tone,
            "language": detect_language(text),
            "entities": entities,
            "text_quality": assess_text_quality(text),
            "call_to_action": detect_call_to_action(text)
        }
        
        logger.info(f"Comprehensive text analysis completed: {len(keywords)} keywords, {tone} tone")
        return result
        
    except Exception as e:
        logger.error(f"Error in comprehensive text analysis: {e}")
        return {
            "sentiment": {"label": "ERROR", "score": 0.0},
            "emotions": {},
            "keywords": [],
            "readability_score": 0.0,
            "tone": "unknown",
            "language": "unknown",
            "entities": [],
            "error": str(e)
        }


def analyze_sentiment(text_pipeline, text: str) -> Dict:
    """Enhanced sentiment analysis with more detailed output"""
    try:
        # Truncate if needed
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        results = text_pipeline(text)
        
        if results and len(results) > 0:
            result = results[0]
            label = result.get("label", "UNKNOWN")
            score = result.get("score", 0.0)
            
            # Map to standard sentiment
            if label == "POSITIVE":
                sentiment_type = "positive"
            elif label == "NEGATIVE":
                sentiment_type = "negative"
            else:
                sentiment_type = "neutral"
            
            return {
                "label": label,
                "type": sentiment_type,
                "score": round(float(score), 4),
                "confidence": "high" if score > 0.8 else "medium" if score > 0.6 else "low"
            }
        
        return {"label": "NEUTRAL", "type": "neutral", "score": 0.5, "confidence": "low"}
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {"label": "ERROR", "type": "unknown", "score": 0.0, "confidence": "none", "error": str(e)}


def detect_emotions(text: str) -> Dict[str, float]:
    """
    Detect emotional content in text using keyword analysis.
    Would use dedicated emotion model in production.
    """
    emotion_keywords = {
        "joy": ["happy", "joy", "excited", "amazing", "wonderful", "great", "love", "excellent"],
        "trust": ["reliable", "trust", "honest", "authentic", "genuine", "certified", "guaranteed"],
        "fear": ["warning", "danger", "risk", "urgent", "limited", "hurry", "last chance"],
        "surprise": ["new", "discover", "reveal", "unveil", "breakthrough", "innovation"],
        "sadness": ["miss", "lose", "end", "final", "last"],
        "anticipation": ["coming soon", "next", "upcoming", "future", "waiting"],
        "anger": ["outrageous", "ridiculous", "unbelievable"],
        "confidence": ["best", "premium", "luxury", "exclusive", "elite", "superior"]
    }
    
    text_lower = text.lower()
    emotions = {}
    
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotions[emotion] = min(score / 3.0, 1.0)  # Normalize
    
    return emotions


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract key terms from text"""
    # Remove punctuation and convert to lowercase
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out common stop words
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be', 
                     'this', 'that', 'these', 'those', 'it', 'its'])
    
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Get most common words
    word_freq = Counter(words)
    keywords = [word for word, _ in word_freq.most_common(top_n)]
    
    return keywords


def calculate_readability(text: str) -> float:
    """Calculate readability score (simplified Flesch Reading Ease)"""
    words = text.split()
    sentences = text.split('.')
    
    if not words or not sentences:
        return 0.0
    
    avg_sentence_length = len(words) / max(len(sentences), 1)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Simplified readability score (0-100, higher = easier to read)
    score = 100 - (avg_sentence_length * 1.5) - (avg_word_length * 10)
    return max(0.0, min(100.0, score))


def analyze_tone(text: str, sentiment: Dict) -> str:
    """Determine overall tone of the text"""
    text_lower = text.lower()
    
    # Check for urgency
    if any(word in text_lower for word in ['now', 'today', 'urgent', 'limited', 'hurry']):
        return "urgent"
    
    # Check for professional
    if any(word in text_lower for word in ['professional', 'business', 'corporate', 'enterprise']):
        return "professional"
    
    # Check for casual/friendly
    if any(word in text_lower for word in ['hey', 'hi', 'hello', '!', 'awesome', 'cool']):
        return "casual"
    
    # Base on sentiment
    if sentiment.get("type") == "positive":
        return "enthusiastic"
    elif sentiment.get("type") == "negative":
        return "serious"
    
    return "neutral"


def detect_language(text: str) -> str:
    """Basic language detection (would use proper library in production)"""
    # Simple heuristic - check for common English words
    english_words = ['the', 'and', 'is', 'to', 'in', 'of', 'for', 'on', 'with']
    text_lower = text.lower()
    
    english_count = sum(1 for word in english_words if word in text_lower)
    
    return "english" if english_count > 2 else "unknown"


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract named entities using pattern matching"""
    entities = []
    
    # Extract URLs
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in urls:
        entities.append({"type": "URL", "value": url})
    
    # Extract emails
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        entities.append({"type": "EMAIL", "value": email})
    
    # Extract phone numbers (basic pattern)
    phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
    for phone in phones:
        entities.append({"type": "PHONE", "value": phone})
    
    # Extract percentages
    percentages = re.findall(r'\d+%', text)
    for pct in percentages:
        entities.append({"type": "PERCENTAGE", "value": pct})
    
    # Extract currency
    currency = re.findall(r'\$\d+(?:\.\d{2})?', text)
    for curr in currency:
        entities.append({"type": "CURRENCY", "value": curr})
    
    return entities


def assess_text_quality(text: str) -> Dict[str, any]:
    """Assess the quality of advertising text"""
    return {
        "length": "optimal" if 10 <= len(text.split()) <= 50 else "suboptimal",
        "has_call_to_action": bool(detect_call_to_action(text)),
        "has_brand_name": bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)),
        "uses_numbers": bool(re.search(r'\d', text)),
        "punctuation_variety": len(set(re.findall(r'[!?.,]', text)))
    }


def detect_call_to_action(text: str) -> bool:
    """Detect if text contains call-to-action phrases"""
    cta_phrases = [
        'buy now', 'shop now', 'learn more', 'get started', 'sign up', 
        'subscribe', 'download', 'try free', 'limited time', 'click here',
        'call now', 'order now', 'book now', 'reserve', 'claim'
    ]
    
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in cta_phrases)

