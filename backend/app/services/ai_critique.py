"""AI-Powered Advertising Critique and Prompt Engineering Service"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def generate_comprehensive_critique(
    ocr_results: Dict,
    text_analysis: Dict,
    image_analysis: Dict,
    user_prompt: str = None
) -> Dict:
    """
    Generate comprehensive advertisement critique based on all analyses.
    
    Args:
        ocr_results: OCR extraction results
        text_analysis: Comprehensive text analysis
        image_analysis: Comprehensive image analysis
        user_prompt: Optional user-provided prompt
        
    Returns:
        Comprehensive critique with scores and recommendations
    """
    try:
        critique = {
            "overall_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "target_audience": "general",
            "emotional_impact": "neutral",
            "effectiveness_rating": "medium"
        }
        
        # Score components
        text_score = evaluate_text_effectiveness(ocr_results, text_analysis)
        visual_score = evaluate_visual_effectiveness(image_analysis)
        composition_score = evaluate_composition(image_analysis)
        
        # Calculate overall score
        critique["overall_score"] = round(
            (text_score * 0.4 + visual_score * 0.35 + composition_score * 0.25),
            2
        )
        
        # Generate strengths
        critique["strengths"] = identify_strengths(text_analysis, image_analysis)
        
        # Generate weaknesses
        critique["weaknesses"] = identify_weaknesses(ocr_results, text_analysis, image_analysis)
        
        # Generate recommendations
        critique["recommendations"] = generate_recommendations(
            critique["weaknesses"],
            text_analysis,
            image_analysis
        )
        
        # Determine target audience
        critique["target_audience"] = determine_target_audience(text_analysis, image_analysis)
        
        # Assess emotional impact
        critique["emotional_impact"] = assess_emotional_impact(text_analysis)
        
        # Rate effectiveness
        critique["effectiveness_rating"] = rate_effectiveness(critique["overall_score"])
        
        logger.info(f"Critique generated: score={critique['overall_score']}, {len(critique['strengths'])} strengths, {len(critique['weaknesses'])} weaknesses")
        return critique
        
    except Exception as e:
        logger.error(f"Error generating critique: {e}")
        return {
            "overall_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "target_audience": "unknown",
            "emotional_impact": "unknown",
            "effectiveness_rating": "unknown",
            "error": str(e)
        }


def generate_master_prompt(
    ocr_results: Dict,
    text_analysis: Dict,
    image_analysis: Dict,
    critique: Dict,
    user_prompt: str = None
) -> str:
    """
    Generate sophisticated master prompt for image generation based on all analyses.
    
    This creates a highly detailed, optimized prompt for Stable Diffusion that incorporates:
    - User's original intent
    - Analyzed visual elements
    - Text sentiment and tone
    - Quality recommendations
    - Professional advertising best practices
    
    Args:
        ocr_results: OCR results
        text_analysis: Text analysis results
        image_analysis: Image analysis results
        critique: Generated critique
        user_prompt: User's original prompt
        
    Returns:
        Master prompt string optimized for Stable Diffusion
    """
    try:
        prompt_parts = []
        
        # 1. Base quality descriptors (critical for SD)
        prompt_parts.append("masterpiece, best quality, highly detailed, professional advertisement,")
        
        # 2. Incorporate user's intent if provided
        if user_prompt and len(user_prompt.strip()) > 0:
            prompt_parts.append(f"{user_prompt.strip()},")
        
        # 3. Visual style based on analysis
        visual_style = determine_visual_style(image_analysis)
        prompt_parts.append(f"{visual_style} style,")
        
        # 4. Color palette (highly important for SD)
        color_description = describe_color_palette(image_analysis.get("colors", {}))
        if color_description:
            prompt_parts.append(f"{color_description},")
        
        # 5. Composition and layout
        composition = describe_composition(image_analysis.get("composition", {}))
        if composition:
            prompt_parts.append(f"{composition},")
        
        # 6. Objects and subjects
        objects_desc = describe_main_subjects(image_analysis.get("objects", {}))
        if objects_desc:
            prompt_parts.append(f"featuring {objects_desc},")
        
        # 7. Emotional tone
        emotional_tone = text_analysis.get("tone", "professional")
        emotions = text_analysis.get("emotions", {})
        if emotions:
            dominant_emotion = max(emotions, key=emotions.get) if emotions else "confident"
            prompt_parts.append(f"{emotional_tone} tone, {dominant_emotion} mood,")
        
        # 8. Text elements if significant
        if ocr_results.get("word_count", 0) > 0:
            text_quality = text_analysis.get("text_quality", {})
            if text_quality.get("has_call_to_action"):
                prompt_parts.append("compelling call-to-action text,")
            prompt_parts.append("clear typography, readable text,")
        
        # 9. Quality enhancers based on critique
        if critique.get("overall_score", 0) < 70:
            # Add enhancement descriptors
            prompt_parts.append("enhanced visual appeal, improved composition,")
            prompt_parts.append("professional color grading, optimized lighting,")
        
        # 10. Technical quality descriptors
        prompt_parts.append("sharp focus, high resolution, 8k quality,")
        prompt_parts.append("professional photography, studio lighting,")
        
        # 11. Negative prompt elements (what to avoid)
        negative_elements = [
            "blurry", "low quality", "amateur", "distorted",
            "ugly", "deformed", "bad anatomy", "worst quality"
        ]
        
        # Join all parts
        master_prompt = " ".join(prompt_parts)
        
        # Optimize length (SD works best with 75-150 tokens)
        master_prompt = optimize_prompt_length(master_prompt)
        
        logger.info(f"Master prompt generated: {len(master_prompt)} characters")
        logger.debug(f"Master prompt: {master_prompt}")
        
        return master_prompt
        
    except Exception as e:
        logger.error(f"Error generating master prompt: {e}")
        return "professional advertisement, high quality, detailed, masterpiece"


# Helper functions

def evaluate_text_effectiveness(ocr_results: Dict, text_analysis: Dict) -> float:
    """Evaluate text effectiveness (0-100)"""
    score = 50.0  # Base score
    
    # Check if text exists
    if ocr_results.get("word_count", 0) > 0:
        score += 10
        
        # Readability
        readability = text_analysis.get("readability_score", 0)
        score += (readability / 100) * 15
        
        # Sentiment
        sentiment = text_analysis.get("sentiment", {})
        if sentiment.get("type") == "positive":
            score += 10
        
        # Call to action
        if text_analysis.get("call_to_action"):
            score += 15
    
    return min(100.0, max(0.0, score))


def evaluate_visual_effectiveness(image_analysis: Dict) -> float:
    """Evaluate visual effectiveness (0-100)"""
    score = 50.0
    
    # Color diversity
    colors = image_analysis.get("colors", {})
    if colors.get("dominant_colors"):
        score += 15
    
    # Object detection
    objects = image_analysis.get("objects", {})
    if objects.get("total_detections", 0) > 0:
        score += 20
    
    # Quality metrics
    quality = image_analysis.get("quality_metrics", {})
    if quality:
        score += 15
    
    return min(100.0, max(0.0, score))


def evaluate_composition(image_analysis: Dict) -> float:
    """Evaluate composition quality (0-100)"""
    score = 60.0  # Base score
    
    composition = image_analysis.get("composition", {})
    if composition:
        score += 20
    
    visual_hierarchy = image_analysis.get("visual_hierarchy", [])
    if visual_hierarchy:
        score += 20
    
    return min(100.0, max(0.0, score))


def identify_strengths(text_analysis: Dict, image_analysis: Dict) -> List[str]:
    """Identify advertisement strengths"""
    strengths = []
    
    # Text strengths
    if text_analysis.get("call_to_action"):
        strengths.append("Strong call-to-action present")
    
    sentiment = text_analysis.get("sentiment", {})
    if sentiment.get("type") == "positive" and sentiment.get("score", 0) > 0.8:
        strengths.append("Highly positive messaging")
    
    readability = text_analysis.get("readability_score", 0)
    if readability > 70:
        strengths.append("Excellent readability")
    
    # Visual strengths
    objects = image_analysis.get("objects", {})
    if objects.get("total_detections", 0) >= 3:
        strengths.append("Rich visual content with multiple elements")
    
    colors = image_analysis.get("colors", {})
    if len(colors.get("dominant_colors", [])) >= 3:
        strengths.append("Vibrant and diverse color palette")
    
    return strengths if strengths else ["Professional presentation"]


def identify_weaknesses(ocr_results: Dict, text_analysis: Dict, image_analysis: Dict) -> List[str]:
    """Identify advertisement weaknesses"""
    weaknesses = []
    
    # Text weaknesses
    if ocr_results.get("word_count", 0) == 0:
        weaknesses.append("No text detected - consider adding key messaging")
    elif ocr_results.get("confidence", 0) < 0.7:
        weaknesses.append("Low OCR confidence - text may be unclear or too small")
    
    if not text_analysis.get("call_to_action"):
        weaknesses.append("Missing clear call-to-action")
    
    readability = text_analysis.get("readability_score", 0)
    if readability < 50:
        weaknesses.append("Text readability could be improved")
    
    # Visual weaknesses
    objects = image_analysis.get("objects", {})
    if objects.get("total_detections", 0) == 0:
        weaknesses.append("No distinct objects detected - image may lack focal points")
    
    return weaknesses if weaknesses else []


def generate_recommendations(weaknesses: List[str], text_analysis: Dict, image_analysis: Dict) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    for weakness in weaknesses:
        if "no text" in weakness.lower():
            recommendations.append("Add concise, impactful headline text")
        elif "call-to-action" in weakness.lower():
            recommendations.append("Include clear CTA like 'Shop Now' or 'Learn More'")
        elif "readability" in weakness.lower():
            recommendations.append("Simplify text or increase font size for better readability")
        elif "no distinct objects" in weakness.lower():
            recommendations.append("Add clear product images or focal elements")
    
    # General best practices
    if len(recommendations) < 3:
        recommendations.append("Consider A/B testing with different color schemes")
        recommendations.append("Ensure brand consistency across visual elements")
    
    return recommendations[:5]  # Limit to top 5


def determine_target_audience(text_analysis: Dict, image_analysis: Dict) -> str:
    """Determine likely target audience"""
    tone = text_analysis.get("tone", "neutral")
    
    if tone == "professional":
        return "business professionals"
    elif tone == "casual":
        return "general consumers"
    elif tone == "urgent":
        return "deal-seeking customers"
    
    return "general audience"


def assess_emotional_impact(text_analysis: Dict) -> str:
    """Assess emotional impact of the advertisement"""
    emotions = text_analysis.get("emotions", {})
    
    if not emotions:
        return "neutral"
    
    dominant_emotion = max(emotions, key=emotions.get)
    emotion_score = emotions[dominant_emotion]
    
    if emotion_score > 0.7:
        return f"strong {dominant_emotion}"
    elif emotion_score > 0.4:
        return f"moderate {dominant_emotion}"
    else:
        return "subtle emotional appeal"


def rate_effectiveness(score: float) -> str:
    """Rate overall effectiveness"""
    if score >= 85:
        return "excellent"
    elif score >= 70:
        return "good"
    elif score >= 55:
        return "moderate"
    else:
        return "needs improvement"


def determine_visual_style(image_analysis: Dict) -> str:
    """Determine visual style for prompt"""
    # This would be more sophisticated with actual image analysis
    return "modern professional advertising"


def describe_color_palette(colors_data: Dict) -> str:
    """Describe color palette for prompt"""
    dominant_colors = colors_data.get("dominant_colors", [])
    
    if not dominant_colors:
        return "vibrant color palette"
    
    # Analyze color warmth
    # In production, would analyze actual hex values
    return "bold and vibrant color scheme"


def describe_composition(composition_data: Dict) -> str:
    """Describe composition for prompt"""
    return "balanced composition, professional layout"


def describe_main_subjects(objects_data: Dict) -> str:
    """Describe main subjects for prompt"""
    objects = objects_data.get("objects", {})
    
    if not objects:
        return "clean minimalist design"
    
    # Get top 3 objects
    sorted_objects = sorted(
        objects.items(),
        key=lambda x: x[1].get("count", 0),
        reverse=True
    )[:3]
    
    subject_names = [obj[0] for obj in sorted_objects if obj[0] != "__background__"]
    
    if subject_names:
        return ", ".join(subject_names)
    
    return "product showcase"


def optimize_prompt_length(prompt: str, max_length: int = 400) -> str:
    """Optimize prompt length for Stable Diffusion"""
    if len(prompt) <= max_length:
        return prompt
    
    # Trim while preserving key information
    parts = prompt.split(",")
    
    # Keep most important parts (first 60% and quality descriptors)
    important_parts = parts[:int(len(parts) * 0.6)]
    
    # Add back quality descriptors if not present
    quality_terms = ["masterpiece", "best quality", "detailed", "professional"]
    for term in quality_terms:
        if not any(term in part.lower() for part in important_parts):
            important_parts.append(term)
    
    optimized = ", ".join(important_parts)
    
    # Final length check
    if len(optimized) > max_length:
        optimized = optimized[:max_length].rsplit(",", 1)[0]
    
    return optimized

