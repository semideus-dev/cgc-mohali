"""AI-Powered Advertising Critique and Prompt Engineering Service"""

import logging
import re
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
    nlp_prompt: str,
    vision_prompt: str,
    critique: Dict,
    user_prompt: str = None
) -> str:
    """
    Generate sophisticated master prompt by combining three specialized prompts.
    
    This is the final prompt composer that intelligently merges:
    1. User's original creative vision (highest priority)
    2. NLP prompt (from OCR + text analysis - textual/emotional insights)
    3. Vision prompt (from TorchVision - visual/compositional insights)
    4. Critique-based enhancements (quality improvements)
    
    Optimized for pollinations.ai which prefers natural, descriptive language.
    
    Args:
        nlp_prompt: Specialized prompt from OCR + NLP analysis
        vision_prompt: Specialized prompt from TorchVision analysis
        critique: Generated critique with recommendations
        user_prompt: User's original creative prompt (optional)
        
    Returns:
        Master prompt string optimized for pollinations.ai (natural language, descriptive)
    """
    try:
        prompt_sections = []
        
        # 1. User's creative vision (highest priority - sets the foundation)
        if user_prompt and len(user_prompt.strip()) > 5:
            user_vision = user_prompt.strip()
            # Ensure it starts with a verb if it doesn't
            if not user_vision.lower().startswith(('create', 'design', 'make', 'generate', 'build')):
                prompt_sections.append(f"Create {user_vision}")
            else:
                prompt_sections.append(user_vision)
        else:
            prompt_sections.append("Create a professional advertisement")
        
        # 2. Add NLP prompt (textual and emotional insights)
        if nlp_prompt and len(nlp_prompt.strip()) > 10:
            # Remove "Advertisement" prefix if present
            nlp_clean = nlp_prompt.replace("Advertisement ", "").strip()
            if nlp_clean:
                prompt_sections.append(nlp_clean)
        
        # 3. Add Vision prompt (visual and compositional insights)
        if vision_prompt and len(vision_prompt.strip()) > 10:
            prompt_sections.append(vision_prompt)
        
        # 4. Apply critique-based enhancements
        enhancement = build_critique_enhancement(critique)
        if enhancement:
            prompt_sections.append(enhancement)
        
        # 5. Add quality baseline
        prompt_sections.append("Ensure professional quality with sharp details and modern aesthetic")
        
        # Join all sections into cohesive prompt
        master_prompt = ". ".join(filter(None, prompt_sections))
        
        # Ensure proper ending
        if not master_prompt.endswith('.'):
            master_prompt += '.'
        
        # Optimize for pollinations.ai (150-400 characters ideal)
        master_prompt = optimize_prompt_for_pollinations(master_prompt)
        
        logger.info(f"Master prompt generated ({len(master_prompt)} chars)")
        logger.info(f"Components - User: {bool(user_prompt)}, NLP: {bool(nlp_prompt)}, Vision: {bool(vision_prompt)}")
        logger.info(f"Final master prompt: {master_prompt[:250]}...")
        
        return master_prompt
        
    except Exception as e:
        logger.error(f"Error generating master prompt: {e}")
        # Intelligent fallback that uses whatever is available
        fallback_parts = []
        if user_prompt:
            fallback_parts.append(user_prompt)
        if nlp_prompt:
            fallback_parts.append(nlp_prompt[:100])
        if vision_prompt:
            fallback_parts.append(vision_prompt[:100])
        
        if fallback_parts:
            return ". ".join(fallback_parts) + ". Make it professional and visually appealing."
        else:
            return "Create a professional, high-quality advertisement with modern design and clear visual hierarchy."


def build_critique_enhancement(critique: Dict) -> str:
    """Build enhancement suggestions from critique for the master prompt."""
    try:
        enhancements = []
        
        overall_score = critique.get("overall_score", 50)
        recommendations = critique.get("recommendations", [])
        
        # If score is moderate or low, add specific enhancements
        if overall_score < 75:
            if overall_score < 50:
                enhancements.append("with significantly enhanced visual appeal and improved clarity")
            else:
                enhancements.append("with refined visual appeal and better clarity")
            
            # Parse top recommendations
            if recommendations and len(recommendations) > 0:
                for rec in recommendations[:2]:
                    rec_lower = rec.lower()
                    if "color" in rec_lower or "contrast" in rec_lower:
                        enhancements.append("optimized color harmony and contrast")
                        break
                    elif "composition" in rec_lower or "layout" in rec_lower:
                        enhancements.append("improved composition and balance")
                        break
        
        # Add target audience context
        target_audience = critique.get("target_audience", "")
        if target_audience and target_audience not in ["unknown", "general"]:
            enhancements.append(f"designed for {target_audience}")
        
        return ", ".join(enhancements) if enhancements else ""
        
    except Exception as e:
        logger.error(f"Error building critique enhancement: {e}")
        return ""


def build_nlp_insights(text_analysis: Dict, ocr_results: Dict) -> str:
    """Build natural language insights from NLP analysis."""
    try:
        insights = []
        
        # Sentiment and emotional tone
        sentiment = text_analysis.get("sentiment", {})
        if sentiment:
            sent_label = sentiment.get("label", "").lower()
            if "positive" in sent_label:
                insights.append("with an uplifting and positive message")
            elif "negative" in sent_label:
                insights.append("with a bold and impactful message")
        
        # Dominant emotion
        emotions = text_analysis.get("emotions", {})
        if emotions:
            dominant_emotion = max(emotions, key=emotions.get)
            if emotions[dominant_emotion] > 0.3:
                emotion_map = {
                    "joy": "joyful and enthusiastic",
                    "excitement": "energetic and exciting",
                    "trust": "trustworthy and reliable",
                    "confidence": "confident and assertive"
                }
                if dominant_emotion in emotion_map:
                    insights.append(f"that feels {emotion_map[dominant_emotion]}")
        
        # Tone
        tone = text_analysis.get("tone", "")
        if tone and tone not in ["neutral", "unknown"]:
            insights.append(f"with a {tone} tone")
        
        # Text hierarchy from OCR
        text_hier = ocr_results.get("text_hierarchy", {})
        if text_hier.get("headline"):
            insights.append(f"featuring prominent headline text")
        
        # Call to action
        if text_analysis.get("text_quality", {}).get("has_call_to_action"):
            insights.append("with a compelling call-to-action")
        
        return ", ".join(insights) if insights else ""
        
    except Exception as e:
        logger.error(f"Error building NLP insights: {e}")
        return ""


def build_vision_insights(image_analysis: Dict) -> str:
    """Build natural language insights from vision analysis."""
    try:
        insights = []
        
        # Main subjects/objects
        objects = image_analysis.get("objects", {})
        if objects and isinstance(objects, dict):
            object_list = objects.get("objects", {})
            if object_list:
                # Get top 3 most prominent objects
                top_objects = sorted(object_list.items(), 
                                   key=lambda x: x[1].get("max_confidence", 0), 
                                   reverse=True)[:3]
                object_names = [obj[0] for obj in top_objects if obj[1].get("max_confidence", 0) > 0.7]
                if object_names:
                    if len(object_names) == 1:
                        insights.append(f"showcasing {object_names[0]}")
                    else:
                        insights.append(f"featuring {', '.join(object_names[:-1])} and {object_names[-1]}")
        
        # Color palette
        colors = image_analysis.get("colors", {})
        if colors:
            color_desc = describe_color_palette(colors)
            if color_desc:
                insights.append(f"with {color_desc}")
        
        # Composition style
        composition = image_analysis.get("composition", {})
        if composition:
            comp_desc = describe_composition(composition)
            if comp_desc:
                insights.append(f"in a {comp_desc} composition")
        
        return ", ".join(insights) if insights else ""
        
    except Exception as e:
        logger.error(f"Error building vision insights: {e}")
        return ""


def build_enhancement_suggestions(critique: Dict) -> str:
    """Build enhancement suggestions from critique."""
    try:
        suggestions = []
        
        overall_score = critique.get("overall_score", 50)
        recommendations = critique.get("recommendations", [])
        
        # If score is low, add enhancements
        if overall_score < 70:
            suggestions.append("with enhanced visual appeal and improved clarity")
            
            # Parse specific recommendations
            if recommendations:
                for rec in recommendations[:2]:  # Top 2 recommendations
                    rec_lower = rec.lower()
                    if "color" in rec_lower:
                        suggestions.append("optimized color harmony")
                    elif "contrast" in rec_lower:
                        suggestions.append("better contrast and readability")
                    elif "composition" in rec_lower:
                        suggestions.append("refined composition")
        
        return ", ".join(suggestions) if suggestions else ""
        
    except Exception as e:
        logger.error(f"Error building enhancement suggestions: {e}")
        return ""


def build_quality_descriptors(critique: Dict, image_analysis: Dict) -> str:
    """Build quality and style descriptors."""
    try:
        descriptors = []
        
        # Professional quality baseline
        descriptors.append("Ensure professional quality")
        
        # Style based on visual analysis
        style = determine_visual_style(image_analysis)
        if style:
            descriptors.append(f"in {style} style")
        
        # Technical quality
        descriptors.append("with sharp details and high clarity")
        
        return ", ".join(descriptors) if descriptors else ""
        
    except Exception as e:
        logger.error(f"Error building quality descriptors: {e}")
        return ""


def build_target_context(critique: Dict, text_analysis: Dict) -> str:
    """Build target audience and purpose context."""
    try:
        context = []
        
        # Target audience
        target_audience = critique.get("target_audience", "")
        if target_audience and target_audience != "unknown":
            context.append(f"designed for {target_audience}")
        
        # Emotional impact goal
        emotional_impact = critique.get("emotional_impact", "")
        if emotional_impact:
            context.append(f"creating {emotional_impact}")
        
        return ", ".join(context) if context else ""
        
    except Exception as e:
        logger.error(f"Error building target context: {e}")
        return ""


def optimize_prompt_for_pollinations(prompt: str) -> str:
    """Optimize prompt for pollinations.ai (prefers natural, descriptive language)."""
    try:
        # Remove redundant commas and clean up
        prompt = re.sub(r',\s*,', ',', prompt)
        prompt = re.sub(r'\s+', ' ', prompt)
        prompt = prompt.strip()
        
        # Pollinations works best with 150-400 characters
        if len(prompt) > 500:
            # Intelligently truncate while keeping key information
            sentences = prompt.split('. ')
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) < 400:
                    truncated.append(sentence)
                    current_length += len(sentence) + 2
                else:
                    break
            
            prompt = '. '.join(truncated)
            if not prompt.endswith('.'):
                prompt += '.'
        
        return prompt
        
    except Exception as e:
        logger.error(f"Error optimizing prompt: {e}")
        return prompt[:400]  # Fallback truncation


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

