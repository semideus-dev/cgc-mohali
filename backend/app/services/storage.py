"""UploadThing Storage Service"""

import io
import logging
from datetime import datetime
from typing import Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Service for handling file uploads to UploadThing"""
    
    def __init__(self):
        """Initialize the UploadThing client with credentials from settings"""
        self.secret = settings.UPLOADTHING_SECRET
        self.app_id = settings.UPLOADTHING_APP_ID
        self.base_url = "https://uploadthing.com/api"
    
    def upload_file(
        self,
        file_bytes: bytes,
        file_name: str,
        content_type: str = "image/png"
    ) -> str:
        """
        Upload a file to UploadThing.
        
        Args:
            file_bytes: The file content as bytes
            file_name: The name/key for the file
            content_type: MIME type of the file
            
        Returns:
            Public URL of the uploaded file
            
        Raises:
            Exception if upload fails
        """
        try:
            # Generate a unique filename with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            unique_file_name = f"{timestamp}_{file_name}"
            
            # Prepare headers
            headers = {
                "X-Uploadthing-Api-Key": self.secret,
                "X-Uploadthing-Version": "6.4.0"
            }
            
            # Prepare the file for upload
            files = {
                "files": (unique_file_name, io.BytesIO(file_bytes), content_type)
            }
            
            # Upload to UploadThing
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/api/uploadFiles",
                    headers=headers,
                    files=files,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    raise Exception(f"UploadThing API error: {response.status_code} - {response.text}")
                
                result = response.json()
                
                # Extract the file URL from response
                if "data" in result and len(result["data"]) > 0:
                    file_url = result["data"][0]["url"]
                    logger.info(f"Successfully uploaded file: {unique_file_name}")
                    return file_url
                else:
                    raise Exception("No file URL returned from UploadThing")
            
        except Exception as e:
            logger.error(f"Failed to upload file to UploadThing: {e}")
            raise Exception(f"Storage upload failed: {str(e)}")
    
    def delete_file(self, file_url: str) -> bool:
        """
        Delete a file from UploadThing.
        
        Args:
            file_url: The URL of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract file key from URL
            # UploadThing URLs typically look like: https://utfs.io/f/{fileKey}
            if "/f/" in file_url:
                file_key = file_url.split("/f/")[-1]
            else:
                logger.warning(f"Could not extract file key from URL: {file_url}")
                return False
            
            headers = {
                "X-Uploadthing-Api-Key": self.secret,
                "X-Uploadthing-Version": "6.4.0"
            }
            
            with httpx.Client() as client:
                response = client.delete(
                    f"{self.base_url}/api/deleteFile",
                    headers=headers,
                    json={"fileKey": file_key},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully deleted file: {file_key}")
                    return True
                else:
                    logger.error(f"Failed to delete file: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting file from UploadThing: {e}")
            return False

