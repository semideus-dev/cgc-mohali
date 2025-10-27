"""Image Generation Service using Stable Diffusion"""

import io
import logging

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def generate_image_with_diffusers(
    pipeline,
    device: torch.device,
    prompt: str,
    image_bytes: bytes,
    num_inference_steps: int = 20,
    strength: float = 0.75
) -> bytes:
    """
    Generate a new image using Stable Diffusion Image-to-Image pipeline.
    
    Args:
        pipeline: The loaded StableDiffusionImg2ImgPipeline
        device: torch.device to run inference on
        prompt: Text prompt for image generation
        image_bytes: Original image bytes to use as base
        num_inference_steps: Number of denoising steps (lower = faster)
        strength: How much to transform the image (0.0-1.0)
        
    Returns:
        Generated image as bytes
    """
    try:
        # Load and prepare the input image
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize to a size compatible with Stable Diffusion (multiple of 8)
        # 512x512 is a good default for SD 1.5
        target_size = (512, 512)
        input_image = input_image.resize(target_size, Image.Resampling.LANCZOS)
        
        logger.info(f"Generating image with prompt: '{prompt[:100]}...'")
        logger.info(f"Steps: {num_inference_steps}, Strength: {strength}")
        
        # Run the pipeline with no gradient computation
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                image=input_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,  # Standard guidance scale
            )
        
        # Get the generated image
        generated_image = result.images[0]
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        generated_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        logger.info("Image generation completed successfully")
        return img_byte_arr.getvalue()
        
    except Exception as e:
        logger.error(f"Error in image generation: {e}")
        # Return original image as fallback
        logger.warning("Returning original image as fallback")
        return image_bytes

