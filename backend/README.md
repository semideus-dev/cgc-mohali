# AdVision Backend

AI-Powered Advertisement Analysis and Generation Platform

## Overview

AdVision analyzes advertisement images using multiple ML models (all running on CPU) to provide comprehensive insights and generate enhanced versions of ads using Stable Diffusion.

### Features

- **Object Detection**: Identifies objects in ads using Faster R-CNN ResNet50
- **Color Analysis**: Extracts dominant colors using KMeans clustering
- **Text Extraction**: Performs OCR using EasyOCR
- **Sentiment Analysis**: Analyzes text sentiment using DistilBERT
- **Image Generation**: Creates enhanced versions using Stable Diffusion Image-to-Image
- **Asynchronous Processing**: Background task processing for long-running ML operations
- **Cloud Storage**: S3-compatible storage for images (Cloudflare R2, AWS S3, etc.)

## Tech Stack

- **Framework**: FastAPI (async)
- **Database**: PostgreSQL (NeonDB)
- **ORM**: SQLAlchemy
- **Migrations**: Alembic
- **ML Models**: All CPU-only
  - Vision: torchvision (Faster R-CNN)
  - Color: OpenCV + scikit-learn
  - OCR: EasyOCR
  - Sentiment: Transformers (DistilBERT)
  - Generation: Diffusers (Stable Diffusion v1.5)

## Project Structure

```
backend/
├── alembic/                    # Database migrations
│   ├── versions/
│   │   └── 001_initial_migration.py
│   ├── env.py
│   └── script.py.mako
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app with model loading
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           └── analysis.py # API endpoints
│   ├── core/
│   │   └── config.py           # Settings management
│   ├── db/
│   │   ├── database.py         # SQLAlchemy setup
│   │   ├── models.py           # ORM models
│   │   └── schemas.py          # Pydantic schemas
│   └── services/
│       ├── ai_vision.py        # Object detection & colors
│       ├── ai_text.py          # OCR & sentiment
│       ├── ai_generator.py     # Image generation
│       ├── storage.py          # S3 file uploads
│       └── processing.py       # Background task orchestrator
├── .env                        # Environment variables (create from .env.example)
├── .env.example                # Example environment variables
├── alembic.ini                 # Alembic configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- PostgreSQL database (or NeonDB account)
- S3-compatible storage (AWS S3, Cloudflare R2, MinIO, etc.)

### 1. Install Dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Using uv (faster):
```bash
uv pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file from the example:
```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
# Database Configuration
DATABASE_URL=postgresql://user:password@host:5432/dbname

# S3-Compatible Storage Configuration
STORAGE_ENDPOINT_URL=https://your-endpoint.com
STORAGE_ACCESS_KEY_ID=your_access_key
STORAGE_SECRET_ACCESS_KEY=your_secret_key
STORAGE_BUCKET_NAME=advision-images
```

#### Setting up NeonDB (Recommended)

1. Go to [neon.tech](https://neon.tech) and create a free account
2. Create a new project
3. Copy the connection string (it looks like: `postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname`)
4. Paste it as `DATABASE_URL` in your `.env` file

#### Setting up Cloudflare R2 (Recommended for storage)

1. Go to Cloudflare Dashboard > R2
2. Create a new bucket
3. Create an API token with R2 permissions
4. Set the environment variables:
   - `STORAGE_ENDPOINT_URL`: Your R2 endpoint (e.g., `https://<account-id>.r2.cloudflarestorage.com`)
   - `STORAGE_ACCESS_KEY_ID`: Your R2 access key
   - `STORAGE_SECRET_ACCESS_KEY`: Your R2 secret key
   - `STORAGE_BUCKET_NAME`: Your bucket name

### 3. Run Database Migrations

Initialize the database with Alembic:

```bash
cd backend
alembic upgrade head
```

This creates the `analysis_jobs` table.

### 4. Start the Server

**Development mode with auto-reload:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Production mode:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Note**: Use only 1 worker because the ML models are loaded into memory. Multiple workers would load models multiple times.

### 5. First Startup (Model Download)

On the first startup, the application will download several ML models:
- **Faster R-CNN** (~160 MB)
- **DistilBERT** (~260 MB)
- **Stable Diffusion v1.5** (~4 GB)

This may take 5-10 minutes depending on your internet connection. Models are cached locally for subsequent runs.

**Expected startup logs:**
```
Starting AdVision Backend - Loading ML Models (CPU-Only)
Using device: cpu
Loading EasyOCR Reader for text extraction...
✓ EasyOCR Reader loaded successfully
Loading DistilBERT sentiment analysis pipeline...
✓ DistilBERT sentiment pipeline loaded successfully
Loading Faster R-CNN ResNet50 FPN for object detection...
✓ Faster R-CNN model loaded successfully
Loading Stable Diffusion Img2Img Pipeline...
✓ Stable Diffusion pipeline loaded successfully
All ML models loaded successfully! API is ready.
```

## API Usage

### Base URL
```
http://localhost:8000
```

### Interactive Documentation
```
http://localhost:8000/docs
```

### Endpoints

#### 1. Upload Image for Analysis

**POST** `/api/v1/analyze`

Upload an advertisement image for analysis.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@/path/to/ad-image.jpg"
```

**Response:** (202 Accepted)
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### 2. Get Analysis Results

**GET** `/api/v1/results/{job_id}`

Retrieve the status and results of an analysis job.

**cURL Example:**
```bash
curl "http://localhost:8000/api/v1/results/123e4567-e89b-12d3-a456-426614174000"
```

**Response (Processing):**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "processing",
  "created_at": "2025-10-27T12:00:00Z",
  "original_image_url": "https://...",
  "results": null
}
```

**Response (Completed):**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "created_at": "2025-10-27T12:00:00Z",
  "original_image_url": "https://storage.../original_ad.jpg",
  "results": {
    "analysis": {
      "objects": {
        "objects": {
          "person": {"count": 2, "max_confidence": 0.95},
          "car": {"count": 1, "max_confidence": 0.87}
        },
        "total_detections": 3
      },
      "colors": {
        "dominant_colors": [
          {"hex": "#3a5f8c", "percentage": 35.2},
          {"hex": "#f2e8d5", "percentage": 28.7}
        ],
        "primary_color": "#3a5f8c"
      },
      "text": {
        "extracted": "Buy Now! Limited Offer",
        "sentiment": {
          "label": "POSITIVE",
          "score": 0.9876
        }
      }
    },
    "critique": {
      "visual_elements": [...],
      "color_scheme": [...],
      "messaging": [...],
      "overall_score": 85.0
    },
    "generated_image_url": "https://storage.../generated_123.png",
    "generation_prompt": "high quality professional advertisement, featuring person, car, vibrant colors, modern design..."
  }
}
```

#### 3. Health Check

**GET** `/health`

Check if the API and all ML models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cpu"
}
```

## How It Works

### Analysis Pipeline

1. **Image Upload**: User uploads an ad image
2. **Storage**: Original image is uploaded to S3-compatible storage
3. **Job Creation**: Database record created with status "processing"
4. **Background Processing**: Async task starts with these steps:
   - **Object Detection**: Faster R-CNN identifies objects in the image
   - **Color Extraction**: KMeans clustering finds dominant colors
   - **Text Extraction**: EasyOCR extracts text from the image
   - **Sentiment Analysis**: DistilBERT analyzes the sentiment of extracted text
   - **Critique Generation**: Synthesizes findings into a structured critique
   - **Prompt Creation**: Generates a text prompt based on analysis
   - **Image Generation**: Stable Diffusion creates an enhanced version
   - **Upload Result**: Generated image uploaded to storage
5. **Status Update**: Job status set to "completed" with all results

### CPU Optimization

All models are explicitly configured to run on CPU:
- PyTorch: `device = torch.device("cpu")`
- Transformers: `device=-1` (forces CPU)
- EasyOCR: `gpu=False`
- Diffusers: `torch_dtype=torch.float32` (CPU compatible)

**Performance Notes:**
- Object detection: ~5-10 seconds
- OCR: ~3-5 seconds
- Sentiment: ~1-2 seconds
- Image generation: ~30-60 seconds (CPU is slow for diffusion models)
- **Total**: ~45-80 seconds per image

## Development

### Running Migrations

Create a new migration:
```bash
alembic revision --autogenerate -m "description"
```

Apply migrations:
```bash
alembic upgrade head
```

Rollback:
```bash
alembic downgrade -1
```

### Running with Auto-Reload

For development, use the `--reload` flag:
```bash
uvicorn app.main:app --reload
```

### Logging

The application uses Python's `logging` module. Logs include:
- Model loading status
- Analysis progress
- Errors and exceptions

## Troubleshooting

### Models Not Loading

**Error**: `Failed to load ML models`

**Solutions**:
1. Ensure you have enough disk space (~5 GB for models)
2. Check your internet connection (first run downloads models)
3. Check the logs for specific model failures

### Database Connection Issues

**Error**: `Connection refused` or `Authentication failed`

**Solutions**:
1. Verify `DATABASE_URL` in `.env` is correct
2. Ensure PostgreSQL is running (or NeonDB is accessible)
3. Test connection: `psql $DATABASE_URL`

### Storage Upload Failures

**Error**: `Storage upload failed`

**Solutions**:
1. Verify all `STORAGE_*` variables in `.env`
2. Ensure bucket exists and is accessible
3. Check bucket permissions (needs write access)

### Slow Image Generation

**Issue**: Stable Diffusion takes 60+ seconds on CPU

**Solutions**:
1. This is expected on CPU (GPU would be 5-10x faster)
2. Reduce `num_inference_steps` in `ai_generator.py` (default: 20)
3. Consider using a smaller model or different architecture

## Production Deployment

### Recommendations

1. **Use a reverse proxy** (Nginx, Caddy) in front of uvicorn
2. **Set worker count to 1** (models are loaded per worker)
3. **Increase server timeout** for long-running image generation
4. **Use environment-specific configs** (separate `.env` files)
5. **Monitor memory usage** (models use ~6-8 GB RAM)
6. **Set up health checks** (`/health` endpoint)

### Example with Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## License

MIT License - See LICENSE file

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues or questions, please open a GitHub issue.

---

Built with ❤️ using FastAPI and PyTorch

