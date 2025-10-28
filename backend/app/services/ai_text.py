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
    Enhanced OCR with multi-pass extraction, spatial analysis, and comprehensive quality metrics.
    
    Args:
        ocr_reader: The loaded EasyOCR Reader instance
        image_bytes: Raw image bytes
        
    Returns:
        Comprehensive dictionary with OCR results, hierarchy, and quality metrics
    """
    try:
        # Convert bytes to numpy array for EasyOCR
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        
        # Multi-pass OCR for better accuracy
        # Pass 1: Standard detection
        results_standard = ocr_reader.readtext(image_array, paragraph=False)
        
        # Pass 2: Paragraph mode for better flow
        results_paragraph = ocr_reader.readtext(image_array, paragraph=True)
        
        # Process and merge results
        extracted_texts = []
        confidences = []
        text_positions = []
        
        for detection in results_standard:
            if len(detection) >= 3:
                bbox, text, confidence = detection[0], detection[1], detection[2]
                if text.strip():
                    extracted_texts.append(text.strip())
                    confidences.append(confidence)
                    
                    # Calculate center position for spatial analysis
                    center_y = (bbox[0][1] + bbox[2][1]) / 2
                    center_x = (bbox[0][0] + bbox[2][0]) / 2
                    text_positions.append({
                        "text": text.strip(),
                        "x": center_x / width,  # Normalize to 0-1
                        "y": center_y / height,
                        "confidence": confidence
                    })
        
        # Join all text
        full_text = " ".join(extracted_texts)
        words = full_text.split()
        
        # Enhanced metrics
        avg_confidence = np.mean(confidences) if confidences else 0.0
        high_conf_texts = [t for t, c in zip(extracted_texts, confidences) if c > 0.7]
        high_conf_percentage = (len(high_conf_texts) / len(extracted_texts) * 100) if extracted_texts else 0
        
        # Text hierarchy analysis (headline, body, caption)
        text_hierarchy = analyze_text_hierarchy(text_positions, full_text)
        
        # Text density and distribution
        text_density = len(full_text) / (width * height) * 1000000  # per megapixel
        
        # Content analysis
        has_numbers = bool(re.search(r'\d', full_text))
        has_special_chars = bool(re.search(r'[!@#$%^&*(),?":{}|<>]', full_text))
        has_urls = bool(re.search(r'http[s]?://|www\.', full_text, re.IGNORECASE))
        has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', full_text))
        
        # Unique words for vocabulary richness
        unique_words = set(word.lower() for word in words if len(word) > 2)
        vocabulary_richness = len(unique_words) / len(words) if words else 0
        
        logger.info(f"Extracted {len(extracted_texts)} segments, confidence: {avg_confidence:.3f}, hierarchy: {text_hierarchy.get('layout_type')}")
        
        return {
            "raw_text": full_text.strip(),
            "segments": extracted_texts,
            "confidence": float(avg_confidence),
            "high_confidence_percentage": round(high_conf_percentage, 2),
            "word_count": len(words),
            "unique_word_count": len(unique_words),
            "char_count": len(full_text),
            "segment_count": len(extracted_texts),
            "text_hierarchy": text_hierarchy,
            "text_density": round(text_density, 4),
            "vocabulary_richness": round(vocabulary_richness, 3),
            "content_markers": {
                "has_numbers": has_numbers,
                "has_special_characters": has_special_chars,
                "has_urls": has_urls,
                "has_email": has_email
            },
            "text_positions": text_positions,
            "readability": {
                "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0,
                "sentence_count": len(re.split(r'[.!?]+', full_text)),
                "avg_sentence_length": len(words) / len(re.split(r'[.!?]+', full_text)) if full_text else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error in text extraction: {e}")
        return {
            "raw_text": "",
            "segments": [],
            "confidence": 0.0,
            "high_confidence_percentage": 0,
            "word_count": 0,
            "unique_word_count": 0,
            "char_count": 0,
            "segment_count": 0,
            "text_hierarchy": {},
            "text_density": 0,
            "vocabulary_richness": 0,
            "content_markers": {},
            "text_positions": [],
            "readability": {},
            "error": str(e)
        }


def analyze_text_hierarchy(text_positions: List[Dict], full_text: str) -> Dict:
    """
    Analyze spatial hierarchy and layout of text (headline, body, caption).
    
    Args:
        text_positions: List of text elements with normalized positions
        full_text: Complete extracted text
        
    Returns:
        Dictionary with text hierarchy and layout analysis
    """
    if not text_positions:
        return {"layout_type": "no_text", "headline": None, "body": [], "caption": None}
    
    try:
        # Sort by vertical position
        sorted_texts = sorted(text_positions, key=lambda x: x['y'])
        
        # Identify layout type based on distribution
        y_positions = [item['y'] for item in text_positions]
        y_spread = max(y_positions) - min(y_positions)
        
        # Determine layout type
        if y_spread < 0.3:
            layout_type = "centered"
        elif len(text_positions) == 1:
            layout_type = "single_element"
        elif len(text_positions) <= 3:
            layout_type = "minimal"
        else:
            layout_type = "structured"
        
        # Heuristic: Top 20% likely headlines, bottom 20% likely captions
        headline = None
        body = []
        caption = None
        
        for item in sorted_texts:
            if item['y'] < 0.2 and headline is None:
                headline = item['text']
            elif item['y'] > 0.8:
                if caption is None:
                    caption = item['text']
            else:
                body.append(item['text'])
        
        # Detect dominant text area (left, center, right)
        x_positions = [item['x'] for item in text_positions]
        avg_x = sum(x_positions) / len(x_positions)
        
        if avg_x < 0.33:
            alignment = "left"
        elif avg_x > 0.66:
            alignment = "right"
        else:
            alignment = "center"
        
        return {
            "layout_type": layout_type,
            "headline": headline,
            "body": body,
            "caption": caption,
            "alignment": alignment,
            "vertical_spread": round(y_spread, 3),
            "has_structured_layout": bool(headline or caption)
        }
        
    except Exception as e:
        logger.error(f"Text hierarchy analysis failed: {e}")
        return {"layout_type": "unknown", "headline": None, "body": [], "caption": None}


def generate_nlp_prompt(ocr_results: Dict, text_analysis: Dict) -> str:
    """
    Generate a specialized prompt from OCR and NLP analysis.
    
    This creates a descriptive prompt focusing on textual elements, sentiment,
    emotional tone, and messaging effectiveness.
    
    Args:
        ocr_results: Enhanced OCR results with hierarchy and metrics
        text_analysis: Comprehensive NLP analysis results
        
    Returns:
        Natural language prompt describing the textual and emotional aspects
    """
    try:
        prompt_parts = []
        
        # 1. Text hierarchy and layout
        text_hier = ocr_results.get("text_hierarchy", {})
        layout_type = text_hier.get("layout_type", "")
        
        if layout_type == "structured":
            prompt_parts.append("with well-structured text layout")
        elif layout_type == "minimal":
            prompt_parts.append("with clean, minimal text")
        elif layout_type == "centered":
            prompt_parts.append("with centrally focused text")
        
        # Add headline if present
        headline = text_hier.get("headline")
        if headline and len(headline) > 3:
            prompt_parts.append(f"featuring the headline '{headline}'")
        
        # 2. Sentiment and emotional tone
        sentiment = text_analysis.get("sentiment", {})
        sentiment_label = sentiment.get("label", "").lower()
        sentiment_score = sentiment.get("score", 0)
        
        if sentiment_score > 0.7:
            if "positive" in sentiment_label:
                prompt_parts.append("conveying an uplifting, positive, and optimistic message")
            elif "negative" in sentiment_label:
                prompt_parts.append("delivering a bold, serious, and impactful message")
        
        # 3. Dominant emotion
        emotions = text_analysis.get("emotions", {})
        if emotions:
            dominant_emotion = max(emotions, key=emotions.get)
            emotion_score = emotions[dominant_emotion]
            
            if emotion_score > 0.3:
                emotion_descriptions = {
                    "joy": "radiating joy and happiness",
                    "excitement": "full of energy and excitement",
                    "trust": "building trust and reliability",
                    "confidence": "projecting confidence and authority",
                    "surprise": "creating surprise and intrigue",
                    "anticipation": "generating anticipation and curiosity"
                }
                if dominant_emotion in emotion_descriptions:
                    prompt_parts.append(emotion_descriptions[dominant_emotion])
        
        # 4. Tone and style
        tone = text_analysis.get("tone", "")
        if tone and tone not in ["neutral", "unknown"]:
            prompt_parts.append(f"with a {tone} and engaging tone")
        
        # 5. Call to action and messaging
        text_quality = text_analysis.get("text_quality", {})
        if text_quality.get("has_call_to_action"):
            prompt_parts.append("including a strong, compelling call-to-action")
        
        # 6. Readability and clarity
        readability = ocr_results.get("readability", {})
        if readability.get("avg_word_length", 0) < 6:
            prompt_parts.append("using clear, easy-to-read language")
        
        # 7. Text density and emphasis
        text_density = ocr_results.get("text_density", 0)
        if text_density > 5:
            prompt_parts.append("with substantial informative text content")
        elif text_density < 2:
            prompt_parts.append("with concise, impactful text")
        
        # 8. Keywords and key themes
        keywords = text_analysis.get("keywords", [])
        if keywords and len(keywords) > 0:
            top_keywords = keywords[:3]
            if top_keywords:
                keyword_str = ", ".join(top_keywords)
                prompt_parts.append(f"emphasizing themes of {keyword_str}")
        
        # Combine into natural language
        if prompt_parts:
            nlp_prompt = "Advertisement " + ", ".join(prompt_parts) + "."
        else:
            nlp_prompt = "Advertisement with professional messaging and clear communication."
        
        logger.info(f"NLP prompt generated: {nlp_prompt[:150]}...")
        return nlp_prompt
        
    except Exception as e:
        logger.error(f"Error generating NLP prompt: {e}")
        return "Advertisement with professional and engaging text content."


def analyze_comprehensive_text(text_pipeline, text: str) -> Dict:
    """
    Enhanced comprehensive NLP analysis with deep semantic understanding and advertising insights.
    
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

