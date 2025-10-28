"""Computer Vision Services: Object Detection, Color Analysis, and Visual Prompt Generation"""

import io
from typing import Dict, List
import logging

import torch
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def generate_vision_prompt(image_analysis: Dict) -> str:
    """
    Generate a specialized prompt from TorchVision analysis.
    
    This creates a descriptive prompt focusing on visual elements, objects,
    colors, composition, and aesthetic qualities.
    
    Args:
        image_analysis: Complete vision analysis results (objects, colors, composition)
        
    Returns:
        Natural language prompt describing the visual and compositional aspects
    """
    try:
        prompt_parts = []
        
        # 1. Main subjects and objects
        objects = image_analysis.get("objects", {})
        if objects and isinstance(objects, dict):
            object_list = objects.get("objects", {})
            
            if object_list:
                # Get top detected objects
                top_objects = sorted(
                    object_list.items(),
                    key=lambda x: x[1].get("count", 0) * x[1].get("max_confidence", 0),
                    reverse=True
                )[:4]
                
                high_conf_objects = [
                    obj[0] for obj in top_objects 
                    if obj[1].get("max_confidence", 0) > 0.6
                ]
                
                if len(high_conf_objects) >= 3:
                    prompt_parts.append(f"Visual composition featuring {high_conf_objects[0]}, {high_conf_objects[1]}, and {high_conf_objects[2]}")
                elif len(high_conf_objects) == 2:
                    prompt_parts.append(f"Showcasing {high_conf_objects[0]} and {high_conf_objects[1]}")
                elif len(high_conf_objects) == 1:
                    prompt_parts.append(f"Prominently displaying {high_conf_objects[0]}")
                else:
                    prompt_parts.append("Clean, minimalist visual composition")
            else:
                prompt_parts.append("Abstract or minimalist visual design")
        
        # 2. Color palette and mood
        colors = image_analysis.get("colors", {})
        if colors:
            dominant_colors = colors.get("dominant_colors", [])
            primary_color = colors.get("primary_color", "")
            
            if dominant_colors and len(dominant_colors) > 0:
                # Analyze color mood
                color_mood = analyze_color_mood(dominant_colors)
                prompt_parts.append(color_mood)
                
                # Describe color scheme
                if len(dominant_colors) == 1:
                    prompt_parts.append(f"with monochromatic {primary_color} color scheme")
                elif len(dominant_colors) == 2:
                    colors_str = " and ".join([c.get("hex", "") for c in dominant_colors[:2]])
                    prompt_parts.append(f"using duotone {colors_str} palette")
                else:
                    color_hexes = [c.get("hex", "") for c in dominant_colors[:3]]
                    prompt_parts.append(f"with vibrant multi-color palette including {', '.join(color_hexes)}")
        
        # 3. Visual quality
        prompt_parts.append("with sharp, professional imagery and clear visual hierarchy")
        
        # Combine into natural language
        if prompt_parts:
            vision_prompt = " ".join(prompt_parts) + "."
        else:
            vision_prompt = "Professional visual design with balanced composition and harmonious colors."
        
        logger.info(f"Vision prompt generated: {vision_prompt[:150]}...")
        return vision_prompt
        
    except Exception as e:
        logger.error(f"Error generating vision prompt: {e}")
        return "Professional, visually appealing design with strong composition."


def analyze_color_mood(dominant_colors: list) -> str:
    """Analyze the emotional mood conveyed by colors."""
    try:
        if not dominant_colors:
            return "with neutral color mood"
        
        # Get primary color hex
        primary_hex = dominant_colors[0].get("hex", "#808080")
        
        # Convert hex to RGB
        hex_clean = primary_hex.lstrip('#')
        r, g, b = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        
        # Analyze color characteristics
        brightness = (r + g + b) / 3
        saturation = (max(r, g, b) - min(r, g, b)) / 255.0 if max(r, g, b) > 0 else 0
        
        # Determine mood
        if saturation > 0.6:
            if r > g and r > b:
                return "with bold, energetic red tones creating excitement"
            elif b > r and b > g:
                return "with cool, trustworthy blue tones inspiring confidence"
            elif g > r and g > b:
                return "with fresh, natural green tones promoting growth"
            elif r > 150 and g > 150:
                return "with warm, optimistic yellow-orange tones radiating positivity"
        elif brightness > 200:
            return "with bright, clean, and modern light tones"
        elif brightness < 80:
            return "with sophisticated, elegant dark tones"
        else:
            return "with balanced, professional color harmony"
            
    except Exception as e:
        logger.error(f"Error analyzing color mood: {e}")
        return "with professional color palette"


def analyze_objects(
    vision_model,
    coco_classes: List[str],
    device: torch.device,
    image_bytes: bytes
) -> Dict:
    """
    Analyze objects in the image using Faster R-CNN.
    
    Args:
        vision_model: The loaded torchvision detection model
        coco_classes: List of COCO class names
        device: torch.device to run inference on
        image_bytes: Raw image bytes
        
    Returns:
        Dictionary containing detected objects and their scores
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Convert to tensor and normalize
        import torchvision.transforms as T
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image).to(device)
        
        # Run inference
        with torch.no_grad():
            predictions = vision_model([image_tensor])
        
        # Filter predictions with score > 0.7
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Filter by confidence threshold
        threshold = 0.7
        high_conf_indices = scores > threshold
        
        filtered_labels = labels[high_conf_indices]
        filtered_scores = scores[high_conf_indices]
        
        # Count objects
        detected_objects = {}
        for label, score in zip(filtered_labels, filtered_scores):
            class_name = coco_classes[label] if label < len(coco_classes) else f"class_{label}"
            if class_name not in detected_objects:
                detected_objects[class_name] = {
                    "count": 0,
                    "max_confidence": 0.0
                }
            detected_objects[class_name]["count"] += 1
            detected_objects[class_name]["max_confidence"] = max(
                detected_objects[class_name]["max_confidence"],
                float(score)
            )
        
        logger.info(f"Detected {len(detected_objects)} unique object types")
        return {
            "objects": detected_objects,
            "total_detections": len(filtered_labels)
        }
        
    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        return {"objects": {}, "total_detections": 0, "error": str(e)}


def analyze_colors(image_bytes: bytes, num_colors: int = 5) -> Dict:
    """
    Extract dominant colors from the image using KMeans clustering.
    
    Args:
        image_bytes: Raw image bytes
        num_colors: Number of dominant colors to extract
        
    Returns:
        Dictionary containing dominant colors as hex codes
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Convert to hex
        hex_colors = []
        for color in colors:
            hex_code = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            hex_colors.append(hex_code)
        
        # Get color distribution (percentage)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = (counts / len(labels) * 100).tolist()
        
        color_analysis = [
            {"hex": hex_colors[i], "percentage": round(percentages[i], 2)}
            for i in range(num_colors)
        ]
        
        # Sort by percentage
        color_analysis.sort(key=lambda x: x["percentage"], reverse=True)
        
        logger.info(f"Extracted {num_colors} dominant colors")
        return {
            "dominant_colors": color_analysis,
            "primary_color": hex_colors[0]
        }
        
    except Exception as e:
        logger.error(f"Error in color analysis: {e}")
        return {"dominant_colors": [], "error": str(e)}

