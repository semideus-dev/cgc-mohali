"""Computer Vision Services: Object Detection and Color Analysis"""

import io
from typing import Dict, List
import logging

import torch
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


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

