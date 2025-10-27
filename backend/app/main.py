"""FastAPI Application Entry Point with ML Model Loading"""

import logging
from contextlib import asynccontextmanager

import torch
import torchvision
from transformers import pipeline
from diffusers import StableDiffusionImg2ImgPipeline
import easyocr

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import analysis
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# COCO class names for object detection
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI.
    
    Startup: Load all ML models into memory (CPU-only)
    Shutdown: Clean up models
    """
    # ==================== STARTUP ====================
    logger.info("=" * 80)
    logger.info("Starting AdVision Backend - Loading ML Models (CPU-Only)")
    logger.info("=" * 80)
    
    # Set device to CPU explicitly
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Store device in app state
    app.state.device = device
    
    try:
        # 1. Load EasyOCR Reader (Text Extraction)
        logger.info("Loading EasyOCR Reader for text extraction...")
        ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        app.state.ocr_reader = ocr_reader
        logger.info("✓ EasyOCR Reader loaded successfully")
        
        # 2. Load Transformers Sentiment Analysis Pipeline (CPU-only)
        logger.info("Loading DistilBERT sentiment analysis pipeline...")
        text_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # -1 forces CPU usage
        )
        app.state.text_pipeline = text_pipeline
        logger.info("✓ DistilBERT sentiment pipeline loaded successfully")
        
        # 3. Load TorchVision Object Detection Model
        logger.info("Loading Faster R-CNN ResNet50 FPN for object detection...")
        vision_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        vision_model.to(device)
        vision_model.eval()  # Set to evaluation mode
        app.state.vision_model = vision_model
        app.state.vision_COCO_CLASSES = COCO_INSTANCE_CATEGORY_NAMES
        logger.info("✓ Faster R-CNN model loaded successfully")
        
        # 4. Load Stable Diffusion Image-to-Image Pipeline (CPU-only)
        logger.info("Loading Stable Diffusion Img2Img Pipeline...")
        logger.info("This may take a few minutes on first run (downloading ~4GB model)...")
        
        diffusers_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,  # Use float32 for CPU
            safety_checker=None,  # Disable safety checker for faster inference
        )
        diffusers_pipeline.to(device)
        
        # Optimize for CPU inference
        # Disable attention slicing and other GPU-specific optimizations
        # diffusers_pipeline.enable_attention_slicing()  # Can help reduce memory on CPU
        
        app.state.diffusers_pipeline = diffusers_pipeline
        logger.info("✓ Stable Diffusion pipeline loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load ML models: {e}", exc_info=True)
        raise
    
    logger.info("=" * 80)
    logger.info("All ML models loaded successfully! API is ready.")
    logger.info("=" * 80)
    
    yield  # Application runs here
    
    # ==================== SHUTDOWN ====================
    logger.info("Shutting down - Cleaning up ML models...")
    
    # Clear all models from memory
    if hasattr(app.state, 'ocr_reader'):
        del app.state.ocr_reader
    if hasattr(app.state, 'text_pipeline'):
        del app.state.text_pipeline
    if hasattr(app.state, 'vision_model'):
        del app.state.vision_model
    if hasattr(app.state, 'diffusers_pipeline'):
        del app.state.diffusers_pipeline
    
    # Clear CUDA cache (won't do anything on CPU but good practice)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Cleanup complete. Goodbye!")


# Create FastAPI application with lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-Powered Advertisement Analysis and Generation Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS - Allow all origins for development/hackathon
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(
    analysis.router,
    prefix=settings.API_V1_STR,
    tags=["analysis"]
)


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "AdVision API Running",
        "status": "healthy",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    models_loaded = all([
        hasattr(app.state, 'ocr_reader'),
        hasattr(app.state, 'text_pipeline'),
        hasattr(app.state, 'vision_model'),
        hasattr(app.state, 'diffusers_pipeline'),
    ])
    
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "device": str(app.state.device) if hasattr(app.state, 'device') else "unknown"
    }

