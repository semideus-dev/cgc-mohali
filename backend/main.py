"""
AdVision Backend Entry Point

This file serves as the entry point for the application.
The actual FastAPI app is defined in app/main.py
"""

from app.main import app

__all__ = ["app"]