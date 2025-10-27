"""Enhanced Image Generation Service using Optimized Stable Diffusion"""

import io
import logging

import torch
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


def generate_image_with_diffusers(
    pipeline,
    device: torch.device,
    prompt: str,
    image_bytes: bytes,
    num_inference_steps: int = 30,  # Increased for better quality
    strength: float = 0.65,  # Balanced transformation
    guidance_scale: float = 8.5,  # Higher for better prompt adherence
    negative_prompt: str = None
) -> bytes:
    """
    Generate enhanced advertisement image using optimized Stable Diffusion parameters.
    
    Args:
        pipeline: The loaded StableDiffusionImg2ImgPipeline
        device: torch.device to run inference on
        prompt: Text prompt for image generation (master prompt)
        image_bytes: Original image bytes to use as base
        num_inference_steps: Number of denoising steps (higher = better quality)
        strength: How much to transform (0.0-1.0, lower preserves more original)
        guidance_scale: How closely to follow prompt (higher = more adherence)
        negative_prompt: What to avoid in generation
        
    Returns:
        Generated image as bytes
    """
    try:
        # Load and prepare the input image
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = input_image.size
        
        # Resize to optimal SD size while preserving aspect ratio
        target_size = calculate_optimal_size(original_size)
        input_image = input_image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Pre-process image for better results
        input_image = enhance_input_image(input_image)
        
        # Default negative prompt if not provided
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, distorted, ugly, amateur, "
                "bad anatomy, deformed, pixelated, grainy, "
                "watermark, text overlay, signature, worst quality"
            )
        
        logger.info(f"Generating enhanced image...")
        logger.info(f"Prompt: '{prompt[:150]}...'")
        logger.info(f"Parameters: steps={num_inference_steps}, strength={strength}, guidance={guidance_scale}")
        
        # Run optimized pipeline
        with torch.no_grad():
            # Enable memory efficient attention if available
            try:
                pipeline.enable_attention_slicing(1)
            except:
                pass
            
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                # Additional quality parameters
                eta=0.0,  # Deterministic generation
            )
        
        # Get the generated image
        generated_image = result.images[0]
        
        # Post-process for enhanced quality
        generated_image = post_process_image(generated_image)
        
        # Resize back to original dimensions if needed
        if original_size != target_size:
            generated_image = generated_image.resize(original_size, Image.Resampling.LANCZOS)
        
        # Convert to bytes with high quality
        img_byte_arr = io.BytesIO()
        generated_image.save(
            img_byte_arr,
            format='PNG',
            optimize=True,
            quality=95
        )
        img_byte_arr.seek(0)
        
        logger.info(f"Image generation completed successfully - Size: {generated_image.size}")
        return img_byte_arr.getvalue()
        
    except Exception as e:
        logger.error(f"Error in image generation: {e}", exc_info=True)
        # Return enhanced original image as fallback
        logger.warning("Generation failed - returning enhanced original image")
        return enhance_fallback_image(image_bytes)


def calculate_optimal_size(original_size: tuple) -> tuple:
    """
    Calculate optimal size for Stable Diffusion (multiple of 64).
    Preserves aspect ratio while staying within reasonable limits.
    """
    width, height = original_size
    aspect_ratio = width / height
    
    # Target around 512-768 pixels on the longer side
    max_dimension = 768
    min_dimension = 512
    
    if width > height:
        new_width = max_dimension
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_dimension
        new_width = int(new_height * aspect_ratio)
    
    # Round to nearest multiple of 64 for SD compatibility
    new_width = ((new_width + 63) // 64) * 64
    new_height = ((new_height + 63) // 64) * 64
    
    # Ensure minimum dimensions
    new_width = max(new_width, min_dimension)
    new_height = max(new_height, min_dimension)
    
    logger.debug(f"Calculated optimal size: {original_size} -> ({new_width}, {new_height})")
    return (new_width, new_height)


def enhance_input_image(image: Image.Image) -> Image.Image:
    """
    Pre-process input image for better SD results.
    """
    try:
        # Slight contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Slight sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        return image
    except Exception as e:
        logger.warning(f"Failed to enhance input image: {e}")
        return image


def post_process_image(image: Image.Image) -> Image.Image:
    """
    Post-process generated image for enhanced quality.
    """
    try:
        # Enhance color vibrancy
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.15)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        # Slight contrast boost
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)
        
        return image
    except Exception as e:
        logger.warning(f"Failed to post-process image: {e}")
        return image


def enhance_fallback_image(image_bytes: bytes) -> bytes:
    """
    Enhance original image as fallback when generation fails.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply enhancements
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.15)
        
        # Save with high quality
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG', quality=95)
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()
    except Exception as e:
        logger.error(f"Failed to enhance fallback image: {e}")
        return image_bytes

